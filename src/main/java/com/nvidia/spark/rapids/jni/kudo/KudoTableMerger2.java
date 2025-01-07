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

import ai.rapids.cudf.BufferType;
import ai.rapids.cudf.HostMemoryBuffer;
import ai.rapids.cudf.Schema;
import com.nvidia.spark.rapids.jni.Arms;
import com.nvidia.spark.rapids.jni.schema.SchemaVisitor2;
import com.nvidia.spark.rapids.jni.schema.Visitors;

import java.util.*;

import static com.nvidia.spark.rapids.jni.Preconditions.ensure;
import static com.nvidia.spark.rapids.jni.kudo.ColumnOffsetInfo.INVALID_OFFSET;
import static com.nvidia.spark.rapids.jni.kudo.KudoSerializer.*;
import static java.lang.Math.*;
import static java.util.Objects.requireNonNull;

/**
 * This class is used to merge multiple KudoTables into a single contiguous buffer, e.g. {@link KudoHostMergeResult},
 * which could be easily converted to a {@link ai.rapids.cudf.ContiguousTable}.
 */
class KudoTableMerger2 implements SchemaVisitor2 {
  // Number of 1s in a byte
  private static final int[] ONES = new int[1024];

  static {
    Arrays.fill(ONES, 0xFFFFFFFF);
  }

  private final List<KudoTable> kudoTables;
  private final List<ColumnOffsetInfo> columnOffsets;
  private final int[] rowCounts;
  private final HostMemoryBuffer buffer;
  private final List<ColumnViewInfo> colViewInfoList;
  private final long[] validityOffsets;
  private final long[] offsetOffsets;
  private final long[] dataOffsets;
  private final Deque<SliceInfo>[] sliceInfos;

  private final SliceInfo[] sliceInfoBuf;
  // A temp buffer for computing
  private final int[] inputBuf;
  private final int[] outputBuf;

  private int curColIdx = 0;
  private KudoHostMergeResult result;

  public KudoTableMerger2(List<KudoTable> tables, HostMemoryBuffer buffer, List<ColumnOffsetInfo> columnOffsets,
                          int[] rowCounts) {
    this.kudoTables = requireNonNull(tables, "tables can't be null");
    requireNonNull(buffer, "buffer can't be null!");
    ensure(columnOffsets != null, "column offsets cannot be null");
    ensure(!columnOffsets.isEmpty(), "column offsets cannot be empty");
    this.columnOffsets = columnOffsets;
    this.rowCounts = rowCounts;
    this.buffer = buffer;
    this.inputBuf = new int[1024];
    this.outputBuf = new int[1024];
    this.colViewInfoList = new ArrayList<>(columnOffsets.size());

    this.validityOffsets = new long[kudoTables.size()];
    this.offsetOffsets = new long[kudoTables.size()];
    this.dataOffsets = new long[kudoTables.size()];

    for (int i = 0; i < kudoTables.size(); i++) {
      validityOffsets[i] = kudoTables.get(i).getHeader().startOffsetOf(BufferType.VALIDITY);
      offsetOffsets[i] = kudoTables.get(i).getHeader().startOffsetOf(BufferType.OFFSET);
      dataOffsets[i] = kudoTables.get(i).getHeader().startOffsetOf(BufferType.DATA);
    }

    sliceInfos = new Deque[kudoTables.size()];
    for (int i = 0; i < sliceInfos.length; i++) {
      sliceInfos[i] = new ArrayDeque<>(8);
      KudoTableHeader header = kudoTables.get(i).getHeader();
      sliceInfos[i].addLast(new SliceInfo(header.getOffset(), header.getNumRows()));
    }

    sliceInfoBuf = new SliceInfo[kudoTables.size()];
  }

  @Override
  public void visitTopSchema(Schema schema) {
    result = new KudoHostMergeResult(schema, buffer, colViewInfoList);
  }

  @Override
  public void visitStruct(Schema structType) {
    ColumnOffsetInfo offsetInfo = getCurColumnOffsets();
    int nullCount = deserializeValidityBuffer(offsetInfo);
    int totalRowCount = rowCounts[curColIdx];
    colViewInfoList.add(new ColumnViewInfo(structType.getType(),
        offsetInfo, nullCount, totalRowCount));


    for (int i=0; i<kudoTables.size(); i++) {
      KudoTableHeader header = kudoTables.get(i).getHeader();
      SliceInfo sliceInfo = sliceInfoOf(i);
      if (header.hasValidityBuffer(curColIdx)) {
        validityOffsets[i] += padForHostAlignment(sliceInfo.getValidityBufferInfo().getBufferLength());
      }
    }
    curColIdx++;
  }

  @Override
  public void preVisitList(Schema listType) {
    ColumnOffsetInfo offsetInfo = getCurColumnOffsets();
    int nullCount = deserializeValidityBuffer(offsetInfo);
    int totalRowCount = rowCounts[curColIdx];
    deserializeOffsetBuffer(offsetInfo);

    colViewInfoList.add(new ColumnViewInfo(listType.getType(),
        offsetInfo, nullCount, totalRowCount));

    for (int i=0; i<kudoTables.size(); i++) {
      KudoTableHeader header = kudoTables.get(i).getHeader();
      SliceInfo sliceInfo = sliceInfoOf(i);
      if (header.hasValidityBuffer(curColIdx)) {
        validityOffsets[i] += padForHostAlignment(sliceInfo.getValidityBufferInfo().getBufferLength());
      }
      if (sliceInfo.getRowCount() > 0) {
        offsetOffsets[i] += padForHostAlignment((sliceInfo.getRowCount() + 1) * Integer.BYTES);
      }
      sliceInfos[i].addLast(sliceInfoBuf[i]);
    }
    curColIdx++;
  }

  @Override
  public void visitList(Schema listType) {
    for (int i = 0; i < kudoTables.size(); i++) {
      sliceInfos[i].removeLast();
    }
  }

  @Override
  public void visit(Schema primitiveType) {
    ColumnOffsetInfo offsetInfo = getCurColumnOffsets();
    int nullCount = deserializeValidityBuffer(offsetInfo);
    int totalRowCount = rowCounts[curColIdx];
    if (primitiveType.getType().hasOffsets()) {
      deserializeOffsetBuffer(offsetInfo);
      deserializeDataBuffer(offsetInfo, OptionalInt.empty());
    } else {
      deserializeDataBuffer(offsetInfo, OptionalInt.of(primitiveType.getType().getSizeInBytes()));
    }

    colViewInfoList.add(new ColumnViewInfo(primitiveType.getType(),
        offsetInfo, nullCount, totalRowCount));

    if (primitiveType.getType().hasOffsets()) {
      for (int i=0; i<kudoTables.size(); i++) {
        KudoTableHeader header = kudoTables.get(i).getHeader();
        SliceInfo sliceInfo = sliceInfoOf(i);
        if (header.hasValidityBuffer(curColIdx)) {
          validityOffsets[i] += padForHostAlignment(sliceInfo.getValidityBufferInfo().getBufferLength());
        }
        if (sliceInfo.getRowCount() > 0) {
          offsetOffsets[i] += padForHostAlignment((sliceInfo.getRowCount() + 1) * Integer.BYTES);
          dataOffsets[i] += padForHostAlignment(sliceInfoBuf[i].getRowCount());
        }
      }
    } else {
      for (int i=0; i<kudoTables.size(); i++) {
        KudoTableHeader header = kudoTables.get(i).getHeader();
        SliceInfo sliceInfo = sliceInfoOf(i);
        if (header.hasValidityBuffer(curColIdx)) {
          validityOffsets[i] += padForHostAlignment(sliceInfo.getValidityBufferInfo().getBufferLength());
        }
        if (sliceInfo.getRowCount() > 0) {
          dataOffsets[i] += padForHostAlignment(primitiveType.getType().getSizeInBytes() * sliceInfo.getRowCount());
        }
      }
    }
    curColIdx++;
  }

  private int deserializeValidityBuffer(ColumnOffsetInfo curColOffset) {
    if (curColOffset.getValidity() != INVALID_OFFSET) {
      int offset = toIntExact(curColOffset.getValidity());
      int nullCountTotal = 0;
      int startRow = 0;
      for (int tableIdx = 0; tableIdx < kudoTables.size(); tableIdx += 1) {
        SliceInfo sliceInfo = sliceInfoOf(tableIdx);
        long validityOffset = validityOffsets[tableIdx];
        if (kudoTables.get(tableIdx).getHeader().hasValidityBuffer(curColIdx)) {
          nullCountTotal += copyValidityBuffer(buffer, offset, startRow,
                  kudoTables.get(tableIdx).getBuffer(),
                  toIntExact(validityOffset),
                  sliceInfo, inputBuf, outputBuf);
        } else {
          appendAllValid(buffer, offset, startRow, sliceInfo.getRowCount());
        }

        startRow += sliceInfo.getRowCount();
      }
      return nullCountTotal;
    } else {
      return 0;
    }
  }

  /**
   * Copy a sliced validity buffer to the destination buffer, starting at the given bit offset.
   *
   * Visible for testing.
   *
   * @return Number of nulls in the validity buffer.
   */
  static int copyValidityBuffer(HostMemoryBuffer dest, int startOffset, int startBit,
                                HostMemoryBuffer src, int srcOffset,
                                SliceInfo sliceInfo, int[] inputBuf, int[] outputBuf) {
    if (sliceInfo.getRowCount() <= 0) {
      return 0;
    }

    int leftRowCount = sliceInfo.getRowCount();
    int nullCount = 0;

    int curDestIntIdx = startOffset + (startBit / 32) * 4;
    int curDestBitIdx = startBit % 32;

    int srcIntBufLen = (sliceInfo.getValidityBufferInfo().getBufferLength() + 3) / 4;
    int curSrcIntIdx = srcOffset;
    int curSrcBitIdx = sliceInfo.getValidityBufferInfo().getBeginBit();

    if (curSrcBitIdx < curDestBitIdx) {
      int rshift = curDestBitIdx - curSrcBitIdx;
      // process first element
      int outputMask = (1 << curDestBitIdx) - 1;
      int destOutput = dest.getInt(curDestIntIdx) & outputMask;

      int rawInput = src.getInt(curSrcIntIdx);
      int input = (rawInput >>> curSrcBitIdx) << curDestBitIdx;
      destOutput = input | destOutput;
      dest.setInt(curDestIntIdx, destOutput);
      int leftRem = max(0, 32 - curDestBitIdx - leftRowCount);
      nullCount += min(leftRowCount, 32 - curDestBitIdx) -  Integer.bitCount((input << leftRem));
      leftRowCount = max(0, leftRowCount - (32 - curDestBitIdx));
      int lastValue = (rawInput & -(1 << curSrcBitIdx)) >>> (32 - rshift);

      curSrcIntIdx += 4;
      curDestIntIdx += 4;
      while (leftRowCount > 0) {
        int curArrLen = min(min(inputBuf.length, outputBuf.length), srcIntBufLen - (curSrcIntIdx - srcOffset) / 4);
        if (curArrLen == 0) {
          dest.setInt(curDestIntIdx, lastValue);
          nullCount += leftRowCount - Integer.bitCount(lastValue & ((1 << leftRowCount) - 1));
          leftRowCount = 0;
        } else {
          src.getInts(inputBuf, 0, curSrcIntIdx, curArrLen);

          for (int i=0; i<curArrLen; i++) {
            outputBuf[i] = (inputBuf[i] << rshift) | lastValue;
            lastValue = inputBuf[i] >>> (32 - rshift);
            // If left count is less than 32, we need to remove most significant bits before counting ones
            leftRem = max(0, 32 - leftRowCount);
            nullCount += min(32, leftRowCount) - Integer.bitCount(outputBuf[i] << leftRem);
            leftRowCount = max(0, leftRowCount - 32);
          }

          dest.setInts(curDestIntIdx, outputBuf, 0, curArrLen);
          curSrcIntIdx += curArrLen * 4;
          curDestIntIdx += curArrLen * 4;
        }
      }
    } else if (curSrcBitIdx > curDestBitIdx) {
      int rshift = curSrcBitIdx - curDestBitIdx;
      int rshiftMask = (1 << rshift) - 1;
      // process first element
      int destOutputMask = (1 << curDestBitIdx) - 1;
      int destOutput = dest.getInt(curDestIntIdx) & destOutputMask;

      int input = src.getInt(curSrcIntIdx);
      int lastValue = (input >>> curSrcBitIdx) << curDestBitIdx;
      destOutput = lastValue | destOutput;
      dest.setInt(curDestIntIdx, destOutput);

      int inputMask = (1 << curSrcBitIdx) - 1;
      int leftRem = max(0, 32 - curSrcBitIdx - leftRowCount);
      nullCount += min(32 - curSrcBitIdx, leftRowCount) - Integer.bitCount((input & ~inputMask) << leftRem);
      leftRowCount = max(0, leftRowCount - (32 - curSrcBitIdx));

      if (leftRowCount == 0) {
        return nullCount;
      }

      curSrcIntIdx += 4;
      int rawInput = src.getInt(curSrcIntIdx);
      input = (rawInput & rshiftMask) << (32 - rshift);
      destOutput |= input;
      dest.setInt(curDestIntIdx, destOutput);
      leftRem = max(0, rshift - leftRowCount);
      nullCount += min(leftRowCount, rshift) - Integer.bitCount(input << leftRem);
      leftRowCount = max(0, leftRowCount - rshift);
      lastValue = rawInput >>> rshift;
      if (leftRowCount == 0) {
        return nullCount;
      }

      curSrcIntIdx += 4;
      curDestIntIdx += 4;
      while (leftRowCount > 0) {
        int curArrLen = min(min(inputBuf.length, outputBuf.length), srcIntBufLen - (curSrcIntIdx - srcOffset) / 4);
        if (curArrLen == 0) {
          dest.setInt(curDestIntIdx, lastValue);
          nullCount += leftRowCount - Integer.bitCount(lastValue & ((1 << leftRowCount) - 1)) ;
          leftRowCount = 0;
        } else {
          src.getInts(inputBuf, 0, curSrcIntIdx, curArrLen);
          for (int i=0; i<curArrLen; i++) {
            input = (inputBuf[i] & rshiftMask) << (32 - rshift);
            outputBuf[i] = input | lastValue;
            lastValue = inputBuf[i] >>> rshift;

            leftRem = max(0, 32 - leftRowCount);
            nullCount += min(32, leftRowCount) - Integer.bitCount(outputBuf[i] << leftRem);
            leftRowCount = max(0, leftRowCount - 32);
          }
          dest.setInts(curDestIntIdx, outputBuf, 0, curArrLen);
          curSrcIntIdx += curArrLen * 4;
          curDestIntIdx += curArrLen * 4;
        }
      }
    } else {
      // Process first element
      int mask = (1 << curDestBitIdx) - 1;
      int firstInput = src.getInt(curSrcIntIdx);
      int destOutput = dest.getInt(curDestIntIdx);
      destOutput = (firstInput & ~mask) | (destOutput & mask);
      dest.setInt(curDestIntIdx, destOutput);
      int leftRem = max(0, 32 - curSrcBitIdx - leftRowCount);
      nullCount += min(32 - curSrcBitIdx, leftRowCount) - Integer.bitCount((firstInput & ~mask) << leftRem);
      leftRowCount = max(0, leftRowCount - (32 - curSrcBitIdx));


      curSrcIntIdx += 4;
      curDestIntIdx += 4;
      while (leftRowCount > 0) {
        int curArrLen = min(min(inputBuf.length, outputBuf.length), srcIntBufLen - (curSrcIntIdx - srcOffset) / 4);
        src.getInts(inputBuf, 0, curSrcIntIdx, curArrLen);
        for (int i=0; i<curArrLen; i++) {
          leftRem = max(0, 32 - leftRowCount);
          nullCount += min(32, leftRowCount) - Integer.bitCount(inputBuf[i] << leftRem);
          leftRowCount = max(0, leftRowCount - 32);
        }
        dest.setInts(curDestIntIdx, inputBuf, 0, curArrLen);
        curSrcIntIdx += curArrLen * 4;
        curDestIntIdx += curArrLen * 4;
      }
    }

    return nullCount;
  }

  static void appendAllValid(HostMemoryBuffer dest, int destOffset, int startBit, int numRows) {
    int curDestIntIdx = destOffset + (startBit / 32) * 4;
    int curDestBitIdx = startBit % 32;

    // First output
    int firstOutput = dest.getInt(curDestIntIdx);
    firstOutput |= -(1 << curDestBitIdx);
    dest.setInt(curDestIntIdx, firstOutput);

    int leftRowCount = max(0, numRows - (32 - curDestBitIdx));

   curDestIntIdx += 4;
    while (leftRowCount > 0) {
      int curArrLen = min(leftRowCount / 32, ONES.length);
      if (curArrLen == 0) {
        dest.setInt(curDestIntIdx, 0xFFFFFFFF);
        leftRowCount = 0;
      } else {
        dest.setInts(curDestIntIdx, ONES, 0, curArrLen);
        leftRowCount = max(0, leftRowCount - 32 * curArrLen);
        curDestIntIdx += curArrLen * 4;
      }
    }
  }

  private void deserializeOffsetBuffer(ColumnOffsetInfo curColOffset) {
    for (int i=0; i<kudoTables.size(); i++) {
      sliceInfoBuf[i] = null;
    }

    if (curColOffset.getOffset() != INVALID_OFFSET) {
      long outputOffset = curColOffset.getOffset();
      HostMemoryBuffer offsetBuf = buffer;

      int accumulatedDataLen = 0;

      for (int tableIdx = 0; tableIdx < kudoTables.size(); tableIdx += 1) {
        SliceInfo sliceInfo = sliceInfoOf(tableIdx);
        if (sliceInfo.getRowCount() > 0) {
          int rowCnt = sliceInfo.getRowCount();

          int firstOffset = offsetOf(tableIdx, 0);
          int lastOffset = offsetOf(tableIdx, rowCnt);
          long inputOffset = offsetOffsets[tableIdx];

          while (rowCnt > 0) {
            int arrLen = min(rowCnt, min(inputBuf.length, outputBuf.length));
            kudoTables.get(tableIdx).getBuffer().getInts(inputBuf, 0, inputOffset, arrLen);

            for (int i = 0; i < arrLen; i++) {
              outputBuf[i] = inputBuf[i] - firstOffset + accumulatedDataLen;
            }

            offsetBuf.setInts(outputOffset, outputBuf, 0, arrLen);
            rowCnt -= arrLen;
            inputOffset += arrLen * Integer.BYTES;
            outputOffset += arrLen * Integer.BYTES;
          }

          sliceInfoBuf[tableIdx] = new SliceInfo(firstOffset, lastOffset - firstOffset);
          accumulatedDataLen += (lastOffset - firstOffset);
        } else {
          sliceInfoBuf[tableIdx] = new SliceInfo(0, 0);
        }
      }

      offsetBuf.setInt(outputOffset, accumulatedDataLen);
    }
  }

  private void deserializeDataBuffer(ColumnOffsetInfo curColOffset, OptionalInt sizeInBytes) {
    if (curColOffset.getData() != INVALID_OFFSET && curColOffset.getDataBufferLen() > 0) {
      long offset = curColOffset.getData();

      if (sizeInBytes.isPresent()) {
        // Fixed size type
        int elementSize = sizeInBytes.getAsInt();

        long start = offset;
        for (int tableIdx = 0; tableIdx < kudoTables.size(); tableIdx += 1) {
          SliceInfo sliceInfo = sliceInfoOf(tableIdx);
          if (sliceInfo.getRowCount() > 0) {
            int thisDataLen = toIntExact(elementSize * sliceInfo.getRowCount());
            copyDataBuffer(buffer, start, tableIdx, thisDataLen);
            start += thisDataLen;
          }
        }
      } else {
        // String type
        long start = offset;
        for (int tableIdx = 0; tableIdx < kudoTables.size(); tableIdx += 1) {
          int thisDataLen = sliceInfoBuf[tableIdx].getRowCount();
          copyDataBuffer(buffer, start, tableIdx, thisDataLen);
          start += thisDataLen;
        }
      }
    }
  }


  private ColumnOffsetInfo getCurColumnOffsets() {
    return columnOffsets.get(curColIdx);
  }

  private SliceInfo sliceInfoOf(int tableIdx) {
    return sliceInfos[tableIdx].getLast();
  }

  private int offsetOf(int tableIdx, long rowIdx) {
    long startOffset = offsetOffsets[tableIdx];
    return kudoTables.get(tableIdx).getBuffer().getInt(startOffset + rowIdx * Integer.BYTES);
  }

  private void copyDataBuffer(HostMemoryBuffer dst, long dstOffset, int tableIdx, int dataLen) {
    long startOffset = dataOffsets[tableIdx];
    dst.copyFromHostBuffer(dstOffset, kudoTables.get(tableIdx).getBuffer(), startOffset, dataLen);
  }

  static KudoHostMergeResult merge(Schema schema, MergedInfoCalc2 mergedInfo) {
    List<KudoTable> serializedTables = mergedInfo.getTables();
    return Arms.closeIfException(HostMemoryBuffer.allocate(mergedInfo.getTotalDataLen(), true),
        buffer -> {
          KudoTableMerger2 merger = new KudoTableMerger2(serializedTables, buffer, mergedInfo.getColumnOffsets(),
                  mergedInfo.getRowCount());
          Visitors.visitSchema(schema, merger);
          return merger.result;
        });
  }
}
