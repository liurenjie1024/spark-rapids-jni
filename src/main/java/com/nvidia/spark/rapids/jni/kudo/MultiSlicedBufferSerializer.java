/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.nvidia.spark.rapids.jni.kudo;

import static com.nvidia.spark.rapids.jni.Preconditions.ensure;
import static com.nvidia.spark.rapids.jni.kudo.KudoSerializer.padForHostAlignment;
import static java.util.Objects.requireNonNull;

import ai.rapids.cudf.BufferType;
import ai.rapids.cudf.DType;
import ai.rapids.cudf.HostColumnVectorCore;
import ai.rapids.cudf.HostMemoryBuffer;
import com.nvidia.spark.rapids.jni.schema.HostColumnsVisitor;
import java.io.IOException;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.List;

/**
 * This class visits a list of columns and serialize one of the buffers (validity, offset, or data) into with kudo
 * format.
 *
 * <p>
 * The host columns are visited in post order, for more details about the visiting process, please refer to
 * {@link HostColumnsVisitor}.
 * </p>
 *
 * <p>
 * For more details about the kudo format, please refer to {@link KudoSerializer}.
 * </p>
 */
class MultiSlicedBufferSerializer implements HostColumnsVisitor<Void> {
  private final List<OutputArgs> outputArgs;
  private final BufferType bufferType;

  private final Deque<SliceInfo>[] sliceInfos;
  private final boolean addCopyBufferTime;
  private final WriteMetrics metrics;

  MultiSlicedBufferSerializer(List<OutputArgs> outputArgs, BufferType bufferType,
                              boolean addCopyBufferTime, WriteMetrics metrics) {
    requireNonNull(outputArgs, "outputArgs is null");
    ensure(!outputArgs.isEmpty(), "outputArgs is empty");
    requireNonNull(metrics, "metrics is null");

    this.outputArgs = outputArgs;
    this.bufferType = bufferType;
    this.sliceInfos = new Deque[outputArgs.size()];
    for (int i = 0; i < outputArgs.size(); i++) {
      this.sliceInfos[i] = new ArrayDeque<>();
      this.sliceInfos[i].addLast(outputArgs.get(i).getSliceInfo());
    }
    this.addCopyBufferTime = addCopyBufferTime;
    this.metrics = metrics;
  }

  @Override
  public Void visitStruct(HostColumnVectorCore col, List<Void> children) {
    for (int i = 0; i < outputArgs.size(); i++) {
      SliceInfo parent = sliceInfos[i].peekLast();

      try {
        switch (bufferType) {
          case VALIDITY:
            this.copySlicedValidity(col, parent, i);
            break;
          case OFFSET:
          case DATA:
            break;
          default:
            throw new IllegalArgumentException("Unexpected buffer type: " + bufferType);
        }

      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }

    return null;
  }

  @Override
  public Void preVisitList(HostColumnVectorCore col) {
    for (int i = 0; i < outputArgs.size(); i++) {
      SliceInfo parent = sliceInfos[i].getLast();


      long bytesCopied = 0;
      try {
        switch (bufferType) {
          case VALIDITY:
            this.copySlicedValidity(col, parent, i);
            break;
          case OFFSET:
            this.copySlicedOffset(col, parent, i);
            break;
          case DATA:
            break;
          default:
            throw new IllegalArgumentException("Unexpected buffer type: " + bufferType);
        }

      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }

    for (int i = 0; i < outputArgs.size(); i++) {
      SliceInfo parent = sliceInfos[i].getLast();
      SliceInfo current;
      if (col.getOffsets() != null) {
        int start = col.getOffsets()
            .getInt(parent.offset * Integer.BYTES);
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
    for (int i = 0; i < outputArgs.size(); i++) {
      sliceInfos[i].removeLast();
    }
    return null;
  }

  @Override
  public Void visit(HostColumnVectorCore col) {
    for (int i = 0; i < outputArgs.size(); i++) {
      SliceInfo parent = sliceInfos[i].getLast();
      try {
        switch (bufferType) {
          case VALIDITY:
            this.copySlicedValidity(col, parent, i);
            break;
          case OFFSET:
            this.copySlicedOffset(col, parent, i);
            break;
          case DATA:
            this.copySlicedData(col, parent, i);
            break;
          default:
            throw new IllegalArgumentException("Unexpected buffer type: " + bufferType);
        }
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }

    return null;
  }

  private void copySlicedValidity(HostColumnVectorCore column, SliceInfo sliceInfo, int outputIdx)
      throws IOException {
    if (column.getValidity() != null && sliceInfo.getRowCount() > 0) {
      HostMemoryBuffer buff = column.getValidity();
      long len = sliceInfo.getValidityBufferInfo().getBufferLength();
      copyBufferAndPadForHost(buff, sliceInfo.getValidityBufferInfo().getBufferOffset(), len,
          outputIdx);
    }
  }

  private void copySlicedOffset(HostColumnVectorCore column, SliceInfo sliceInfo, int outputIdx)
      throws IOException {
    if (sliceInfo.rowCount <= 0 || column.getOffsets() == null) {
      // Don't copy anything, there are no rows
      return;
    }
    long bytesToCopy = (sliceInfo.rowCount + 1) * Integer.BYTES;
    long srcOffset = sliceInfo.offset * Integer.BYTES;
    copyBufferAndPadForHost(column.getOffsets(), srcOffset, bytesToCopy, outputIdx);
  }

  private void copySlicedData(HostColumnVectorCore column, SliceInfo sliceInfo, int outputIdx)
      throws IOException {
    if (sliceInfo.rowCount > 0) {
      DType type = column.getType();
      if (type.equals(DType.STRING)) {
        long startByteOffset = column.getOffsets().getInt(sliceInfo.offset * Integer.BYTES);
        long endByteOffset =
            column.getOffsets().getInt((sliceInfo.offset + sliceInfo.rowCount) * Integer.BYTES);
        long bytesToCopy = endByteOffset - startByteOffset;
        if (column.getData() == null) {
          if (bytesToCopy != 0) {
            throw new IllegalStateException("String column has no data buffer, " +
                "but bytes to copy is not zero: " + bytesToCopy);
          }
        } else {
          copyBufferAndPadForHost(column.getData(), startByteOffset, bytesToCopy, outputIdx);
        }
      } else if (type.getSizeInBytes() > 0) {
        long bytesToCopy = sliceInfo.rowCount * type.getSizeInBytes();
        long srcOffset = sliceInfo.offset * type.getSizeInBytes();
        copyBufferAndPadForHost(column.getData(), srcOffset, bytesToCopy, outputIdx);
      }
    }
  }

  private void copyBufferAndPadForHost(HostMemoryBuffer buffer, long offset, long length,
                                       int outputIdx)
      throws IOException {
    DataWriter writer = outputArgs.get(outputIdx).getDataWriter();
    if (addCopyBufferTime) {
      long now = System.nanoTime();
      writer.copyDataFrom(buffer, offset, length);
      long ret = padForHostAlignment(writer, length);
      metrics.addCopyBufferTime(System.nanoTime() - now);
      metrics.addWrittenBytes(ret);
    } else {
      writer.copyDataFrom(buffer, offset, length);
      metrics.addWrittenBytes(padForHostAlignment(writer, length));
    }
  }
}
