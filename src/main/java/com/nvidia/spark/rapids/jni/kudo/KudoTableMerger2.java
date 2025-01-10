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
  private static final int[] NUMBER_OF_ONES = new int[256];

  static {
    for (int i = 0; i < NUMBER_OF_ONES.length; i += 1) {
      int count = 0;
      for (int j = 0; j < 8; j += 1) {
        if ((i & (1 << j)) != 0) {
          count += 1;
        }
      }
      NUMBER_OF_ONES[i] = count;
    }
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
      sliceInfos[i] = new LinkedList<>();
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
      SliceInfo sliceInfo = sliceInfoOf(i);
      validityOffsets[i] += padForHostAlignment(sliceInfo.getValidityBufferInfo().getBufferLength());
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
      SliceInfo sliceInfo = sliceInfoOf(i);
      validityOffsets[i] += padForHostAlignment(sliceInfo.getValidityBufferInfo().getBufferLength());
      offsetOffsets[i] += padForHostAlignment(sliceInfo.getRowCount() * Integer.BYTES);
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
        SliceInfo sliceInfo = sliceInfoOf(i);
        validityOffsets[i] += padForHostAlignment(sliceInfo.getValidityBufferInfo().getBufferLength());
        offsetOffsets[i] += padForHostAlignment(sliceInfo.getRowCount() * Integer.BYTES);
        dataOffsets[i] += padForHostAlignment(sliceInfoBuf[i].getRowCount());
      }
    } else {
      for (int i=0; i<kudoTables.size(); i++) {
        SliceInfo sliceInfo = sliceInfoOf(i);
        validityOffsets[i] += padForHostAlignment(sliceInfo.getValidityBufferInfo().getBufferLength());
        dataOffsets[i] += padForHostAlignment(primitiveType.getType().getSizeInBytes() * sliceInfo.getRowCount());
      }
    }
    curColIdx++;
  }

  private int deserializeValidityBuffer(ColumnOffsetInfo curColOffset) {
    if (curColOffset.getValidity() != INVALID_OFFSET) {
      long offset = curColOffset.getValidity();
      long validityBufferSize = curColOffset.getValidityBufferLen();
      try (HostMemoryBuffer validityBuffer = buffer.slice(offset, validityBufferSize)) {
        int nullCountTotal = 0;
        int startRow = 0;
        for (int tableIdx = 0; tableIdx < kudoTables.size(); tableIdx += 1) {
          SliceInfo sliceInfo = sliceInfoOf(tableIdx);
          long validityOffset = validityOffsets[tableIdx];
          if (validityOffset != INVALID_OFFSET) {
            nullCountTotal += copyValidityBuffer(validityBuffer, startRow,
                kudoTables.get(tableIdx).getBuffer(),
                 toIntExact(validityOffset),
                sliceInfo);
          } else {
            appendAllValid(validityBuffer, startRow, sliceInfo.getRowCount());
          }

          System.out.println("nullCountTotal: " + nullCountTotal + "cur col idx: " + curColIdx + " table idx: " + tableIdx);
          startRow += sliceInfo.getRowCount();
        }
        return nullCountTotal;
      }
    } else {
      return 0;
    }
  }

  /**
   * Copy a sliced validity buffer to the destination buffer, starting at the given bit offset.
   *
   * @return Number of nulls in the validity buffer.
   */
  private static int copyValidityBuffer(HostMemoryBuffer dest, int startBit,
                                        HostMemoryBuffer src, int srcOffset,
                                        SliceInfo sliceInfo) {
    int nullCount = 0;
    int totalRowCount = sliceInfo.getRowCount();
    int curIdx = 0;
    int curSrcByteIdx = srcOffset;
    int curSrcBitIdx = sliceInfo.getValidityBufferInfo().getBeginBit();
    int curDestByteIdx = startBit / 8;
    int curDestBitIdx = startBit % 8;

    while (curIdx < totalRowCount) {
      int leftRowCount = totalRowCount - curIdx;
      int appendCount;
      if (curDestBitIdx == 0) {
        appendCount = min(8, leftRowCount);
      } else {
        appendCount = min(8 - curDestBitIdx, leftRowCount);
      }

      int leftBitsInCurSrcByte = 8 - curSrcBitIdx;
      byte srcByte = src.getByte(curSrcByteIdx);
      if (leftBitsInCurSrcByte >= appendCount) {
        // Extract appendCount bits from srcByte, starting from curSrcBitIdx
        byte mask = (byte) (((1 << appendCount) - 1) & 0xFF);
        srcByte = (byte) ((srcByte >>> curSrcBitIdx) & mask);

        nullCount += (appendCount - NUMBER_OF_ONES[srcByte & 0xFF]);

        // Sets the bits in destination buffer starting from curDestBitIdx to 0
        byte destByte = dest.getByte(curDestByteIdx);
        destByte = (byte) (destByte & ((1 << curDestBitIdx) - 1) & 0xFF);

        // Update destination byte with the bits from source byte
        destByte = (byte) ((destByte | (srcByte << curDestBitIdx)) & 0xFF);
        dest.setByte(curDestByteIdx, destByte);

        curSrcBitIdx += appendCount;
        if (curSrcBitIdx == 8) {
          curSrcBitIdx = 0;
          curSrcByteIdx += 1;
        }
      } else {
        // Extract appendCount bits from srcByte, starting from curSrcBitIdx
        byte mask = (byte) (((1 << leftBitsInCurSrcByte) - 1) & 0xFF);
        srcByte = (byte) ((srcByte >>> curSrcBitIdx) & mask);

        byte nextSrcByte = src.getByte(curSrcByteIdx + 1);
        byte nextSrcByteMask = (byte) ((1 << (appendCount - leftBitsInCurSrcByte)) - 1);
        nextSrcByte = (byte) (nextSrcByte & nextSrcByteMask);
        nextSrcByte = (byte) (nextSrcByte << leftBitsInCurSrcByte);
        srcByte = (byte) (srcByte | nextSrcByte);

        nullCount += (appendCount - NUMBER_OF_ONES[srcByte & 0xFF]);

        // Sets the bits in destination buffer starting from curDestBitIdx to 0
        byte destByte = dest.getByte(curDestByteIdx);
        destByte = (byte) (destByte & ((1 << curDestBitIdx) - 1));

        // Update destination byte with the bits from source byte
        destByte = (byte) (destByte | (srcByte << curDestBitIdx));
        dest.setByte(curDestByteIdx, destByte);

        // Update the source byte index and bit index
        curSrcByteIdx += 1;
        curSrcBitIdx = appendCount - leftBitsInCurSrcByte;
      }

      curIdx += appendCount;

      // Update the destination byte index and bit index
      curDestBitIdx += appendCount;
      if (curDestBitIdx == 8) {
        curDestBitIdx = 0;
        curDestByteIdx += 1;
      }
    }

    return nullCount;
  }

  private static void appendAllValid(HostMemoryBuffer dest, int startBit, int numRowsLong) {
    int numRows = toIntExact(numRowsLong);
    int curDestByteIdx = startBit / 8;
    int curDestBitIdx = startBit % 8;

    if (curDestBitIdx > 0) {
      int numBits = 8 - curDestBitIdx;
      int mask = ((1 << numBits) - 1) << curDestBitIdx;
      dest.setByte(curDestByteIdx, (byte) (dest.getByte(curDestByteIdx) | mask));
      curDestByteIdx += 1;
      numRows -= numBits;
    }

    if (numRows > 0) {
      int numBytes = (numRows + 7) / 8;
      dest.setMemory(curDestByteIdx, numBytes, (byte) 0xFF);
    }
  }

  private void deserializeOffsetBuffer(ColumnOffsetInfo curColOffset) {
    if (curColOffset.getOffset() != INVALID_OFFSET) {
      long offset = curColOffset.getOffset();
      HostMemoryBuffer offsetBuf = buffer;

      int accumulatedDataLen = 0;

      for (int tableIdx = 0; tableIdx < kudoTables.size(); tableIdx += 1) {
        SliceInfo sliceInfo = sliceInfoOf(tableIdx);
        if (sliceInfo.getRowCount() > 0) {
          int rowCnt = sliceInfo.getRowCount();

          int firstOffset = offsetOf(tableIdx, 0);
          int lastOffset = offsetOf(tableIdx, rowCnt);

          for (int i = 0; i < rowCnt; i += 1) {
            offsetBuf.setInt(offset, offsetOf(tableIdx, i) - firstOffset + accumulatedDataLen);
            offset += Integer.BYTES;
          }

          sliceInfoBuf[curColIdx] = new SliceInfo(firstOffset, lastOffset - firstOffset);
          accumulatedDataLen += (lastOffset - firstOffset);
        }
      }

      offsetBuf.setInt(offset, accumulatedDataLen);
    }
  }

  private void deserializeDataBuffer(ColumnOffsetInfo curColOffset, OptionalInt sizeInBytes) {
    if (curColOffset.getData() != INVALID_OFFSET && curColOffset.getDataBufferLen() > 0) {
      long offset = curColOffset.getData();
      long dataLen = curColOffset.getDataBufferLen();

      try (HostMemoryBuffer buf = buffer.slice(offset, dataLen)) {
        if (sizeInBytes.isPresent()) {
          // Fixed size type
          int elementSize = sizeInBytes.getAsInt();

          long start = 0;
          for (int tableIdx = 0; tableIdx < kudoTables.size(); tableIdx += 1) {
            SliceInfo sliceInfo = sliceInfoOf(tableIdx);
            if (sliceInfo.getRowCount() > 0) {
              int thisDataLen = toIntExact(elementSize * sliceInfo.getRowCount());
              copyDataBuffer(buf, start, tableIdx, thisDataLen);
              start += thisDataLen;
            }
          }
        } else {
          // String type
          long start = 0;
          for (int tableIdx = 0; tableIdx < kudoTables.size(); tableIdx += 1) {
            int thisDataLen = sliceInfoBuf[curColIdx].getRowCount();
            copyDataBuffer(buf, start, tableIdx, thisDataLen);
            start += thisDataLen;
          }
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
    return Arms.closeIfException(HostMemoryBuffer.allocate(mergedInfo.getTotalDataLen()),
        buffer -> {
          KudoTableMerger2 merger = new KudoTableMerger2(serializedTables, buffer, mergedInfo.getColumnOffsets(),
                  mergedInfo.getRowCount());
          Visitors.visitSchema(schema, merger);
          return merger.result;
        });
  }
}
