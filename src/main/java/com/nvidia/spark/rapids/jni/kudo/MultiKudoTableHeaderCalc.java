package com.nvidia.spark.rapids.jni.kudo;

import static com.nvidia.spark.rapids.jni.Preconditions.ensure;
import static com.nvidia.spark.rapids.jni.kudo.KudoSerializer.padForHostAlignment;
import static java.lang.Math.toIntExact;
import static java.util.Objects.requireNonNull;

import ai.rapids.cudf.DType;
import ai.rapids.cudf.HostColumnVectorCore;
import com.nvidia.spark.rapids.jni.schema.HostColumnsVisitor;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.List;

public class MultiKudoTableHeaderCalc implements HostColumnsVisitor<Void> {
  private final int numFlattenedCols;
  private final byte[] bitset;
  private final List<OutputArgs> outputArgs;
  private long[] validityBufferLen;
  private long[] offsetBufferLen;
  private long[] totalDataLen;
  private int nextColIdx;

  private Deque<SliceInfo>[] sliceInfos;


  MultiKudoTableHeaderCalc(List<OutputArgs> outputArgs,  int numFlattenedCols) {
    requireNonNull(outputArgs, "outputArgs is null");
    ensure(!outputArgs.isEmpty(), "outputArgs is empty");
    this.outputArgs = outputArgs;
    this.totalDataLen = new long[outputArgs.size()];
    this.sliceInfos = new Deque[outputArgs.size()];
    for (int i = 0; i < outputArgs.size() ; i++) {
      this.sliceInfos[i] = new ArrayDeque<>();
      this.sliceInfos[i].addLast(outputArgs.get(i).getSliceInfo());
    }
    this.bitset = new byte[(numFlattenedCols + 7) / 8];
    this.numFlattenedCols = numFlattenedCols;
    this.nextColIdx = 0;
  }

  public List<KudoTableHeader> getHeaders() {
    List<KudoTableHeader> headers = new java.util.ArrayList<>(outputArgs.size());
    for (int i=0; i<outputArgs.size(); i++) {
      SliceInfo root = outputArgs.get(i).getSliceInfo();
      headers.add(new KudoTableHeader(toIntExact(root.offset),
          toIntExact(root.rowCount),
          toIntExact(validityBufferLen[i]),
          toIntExact(offsetBufferLen[i]),
          toIntExact(totalDataLen[i]),
          numFlattenedCols,
          bitset));
    }

    return headers;
  }

  @Override
  public Void visitStruct(HostColumnVectorCore col, List<Void> children) {
    for (int i=0; i<outputArgs.size(); i++) {
      SliceInfo parent = sliceInfos[i].getLast();

      long validityBufferLength = 0;
      if (col.hasValidityVector()) {
        validityBufferLength = padForHostAlignment(parent.getValidityBufferInfo().getBufferLength());
      }

      this.validityBufferLen[i] += validityBufferLength;

      this.totalDataLen[i] += validityBufferLength;
    }
    this.setHasValidity(col.hasValidityVector());
    return null;
  }

  @Override
  public Void preVisitList(HostColumnVectorCore col) {
    for (int i=0; i<outputArgs.size(); i++) {
      SliceInfo parent = sliceInfos[i].getLast();

      long validityBufferLength = 0;
      if (col.hasValidityVector() && parent.rowCount > 0) {
        validityBufferLength = padForHostAlignment(parent.getValidityBufferInfo().getBufferLength());
      }


      long offsetBufferLength = 0;
      if (col.getOffsets() != null && parent.rowCount > 0) {
        offsetBufferLength = padForHostAlignment((parent.rowCount + 1) * Integer.BYTES);
      }

      this.validityBufferLen[i] += validityBufferLength;
      this.offsetBufferLen[i] += offsetBufferLength;
      this.totalDataLen[i] += validityBufferLength + offsetBufferLength;
    }

    this.setHasValidity(col.hasValidityVector());

    for (int i=0; i<outputArgs.size(); i++) {
      SliceInfo parent = sliceInfos[i].getLast();

      SliceInfo current;
      if (col.getOffsets() != null) {
        int start = col.getOffsets().getInt(parent.offset * Integer.BYTES);
        int end = col.getOffsets().getInt((parent.offset + parent.rowCount) * Integer.BYTES);
        int rowCount = end - start;
        current = new SliceInfo(start, rowCount);
      } else {
        current = new SliceInfo(0, 0);
      }

      sliceInfos[i].addLast(current);
    }

    return null;
  }

  @Override
  public Void visitList(HostColumnVectorCore col, Void preVisitResult, Void childResult) {
    for (int i=0; i<outputArgs.size(); i++) {
      sliceInfos[i].removeLast();
    }

    return null;
  }


  @Override
  public Void visit(HostColumnVectorCore col) {
    for (int i = 0; i < outputArgs.size(); i++) {
      SliceInfo parent = sliceInfos[i].peekLast();
      long validityBufferLen = dataLenOfValidityBuffer(col, parent);
      this.validityBufferLen[i] += validityBufferLen;
      this.totalDataLen[i] += validityBufferLen;

      this.setHasValidity(col.hasValidityVector());
    }

    for (int i=0; i<outputArgs.size(); i++) {
      SliceInfo parent = sliceInfos[i].peekLast();
      long offsetBufferLen = dataLenOfOffsetBuffer(col, parent);
      this.offsetBufferLen[i] += offsetBufferLen;
      this.totalDataLen[i] += offsetBufferLen;
    }

    for (int i=0; i<outputArgs.size(); i++) {
      SliceInfo parent = sliceInfos[i].peekLast();
      long dataBufferLen = dataLenOfDataBuffer(col, parent);
      this.totalDataLen[i] += dataBufferLen;
    }

    this.setHasValidity(col.hasValidityVector());
    return null;
  }

  private void setHasValidity(boolean hasValidityBuffer) {
    if (hasValidityBuffer) {
      int bytePos = nextColIdx / 8;
      int bitPos = nextColIdx % 8;
      bitset[bytePos] = (byte) (bitset[bytePos] | (1 << bitPos));
    }
    nextColIdx++;
  }

  private static long dataLenOfValidityBuffer(HostColumnVectorCore col, SliceInfo info) {
    if (col.hasValidityVector() && info.getRowCount() > 0) {
      return padForHostAlignment(info.getValidityBufferInfo().getBufferLength());
    } else {
      return 0;
    }
  }

  private static long dataLenOfOffsetBuffer(HostColumnVectorCore col, SliceInfo info) {
    if (DType.STRING.equals(col.getType()) && info.getRowCount() > 0) {
      return padForHostAlignment((info.rowCount + 1) * Integer.BYTES);
    } else {
      return 0;
    }
  }

  private static long dataLenOfDataBuffer(HostColumnVectorCore col, SliceInfo info) {
    if (DType.STRING.equals(col.getType())) {
      if (col.getOffsets() != null) {
        long startByteOffset = col.getOffsets().getInt(info.offset * Integer.BYTES);
        long endByteOffset = col.getOffsets().getInt((info.offset + info.rowCount) * Integer.BYTES);
        return padForHostAlignment(endByteOffset - startByteOffset);
      } else {
        return 0;
      }
    } else {
      if (col.getType().getSizeInBytes() > 0) {
        return padForHostAlignment(col.getType().getSizeInBytes() * info.rowCount);
      } else {
        return 0;
      }
    }
  }
}
