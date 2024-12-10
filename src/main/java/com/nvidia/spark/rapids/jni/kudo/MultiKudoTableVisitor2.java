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
import static com.nvidia.spark.rapids.jni.kudo.ColumnOffsetInfo.INVALID_OFFSET;
import static com.nvidia.spark.rapids.jni.kudo.KudoSerializer.padForHostAlignment;
import static java.lang.Math.toIntExact;

import ai.rapids.cudf.HostMemoryBuffer;
import ai.rapids.cudf.Schema;
import com.nvidia.spark.rapids.jni.schema.SchemaVisitor;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Deque;
import java.util.List;
import java.util.Objects;

/**
 * This class provides a base class for visiting multiple kudo tables, e.g. it helps to maintain internal states during
 * visiting multi kudo tables, which makes it easier to do some calculations based on them.
 * <br/>
 * The schema used when visiting these kudo tables must be same as the schema used when creating these kudo tables.
 */
abstract class MultiKudoTableVisitor2<T, P, R> implements SchemaVisitor<T, P, R> {
  private final List<KudoTable> tables;
  private final long[] currentColumnOffsets;
  private final Deque<SliceInfo>[] sliceInfoStack;
  private final Deque<Integer> totalRowCountStack;
  // A temporary variable to keep if current column has null
  private boolean hasNull;
  private int currentIdx;
  // Temporary buffer to store data length of string column to avoid repeated allocation
  private final int[] strDataLen;
  // Temporary variable to calculate total data length of string column
  private long totalStrDataLen;

  protected MultiKudoTableVisitor2(List<KudoTable> inputTables) {
    Objects.requireNonNull(inputTables, "tables cannot be null");
    ensure(!inputTables.isEmpty(), "tables cannot be empty");
    this.tables = inputTables instanceof ArrayList ? inputTables : new ArrayList<>(inputTables);
    this.currentColumnOffsets = new long[tables.size()];
    this.sliceInfoStack = new Deque[tables.size()];
    long totalRowCount = 0L;
    for (int i = 0; i < tables.size(); i++) {
      this.currentColumnOffsets[i] = 0;
      KudoTableHeader header = tables.get(i).getHeader();
      this.sliceInfoStack[i] = new ArrayDeque<>(16);
      this.sliceInfoStack[i].add(new SliceInfo(header.getOffset(), header.getNumRows()));
      totalRowCount += header.getNumRows();
    }
    this.totalRowCountStack = new ArrayDeque<>(16);
    totalRowCountStack.addLast(toIntExact(totalRowCount));
    this.hasNull = true;
    this.currentIdx = 0;
    this.strDataLen = new int[tables.size()];
    this.totalStrDataLen = 0;
  }

  List<KudoTable> getTables() {
    return tables;
  }

  @Override
  public R visitTopSchema(Schema schema, List<T> children) {
    return doVisitTopSchema(schema, children);
  }

  protected abstract R doVisitTopSchema(Schema schema, List<T> children);

  @Override
  public T visitStruct(Schema structType, List<T> children) {
    updateHasNull();
    T t = doVisitStruct(structType, children);
    updateOffsets(
        false, // Update offset buffer offset
        false, // Update data buffer offset
        false, // Update slice info
        -1 // element size in bytes, not used for struct
    );
    currentIdx += 1;
    return t;
  }

  protected abstract T doVisitStruct(Schema structType, List<T> children);

  @Override
  public P preVisitList(Schema listType) {
    updateHasNull();
    P t = doPreVisitList(listType);
    updateOffsets(
        true, // update offset buffer offset
        false, // update data buffer offset
        true, // update slice info
        Integer.BYTES // element size in bytes
    );
    currentIdx += 1;
    return t;
  }

  protected abstract P doPreVisitList(Schema listType);

  @Override
  public T visitList(Schema listType, P preVisitResult, T childResult) {
    T t = doVisitList(listType, preVisitResult, childResult);
    for (int tableIdx = 0; tableIdx < tables.size(); tableIdx++) {
      sliceInfoStack[tableIdx].removeLast();
    }
    totalRowCountStack.removeLast();
    return t;
  }

  protected abstract T doVisitList(Schema listType, P preVisitResult, T childResult);

  @Override
  public T visit(Schema primitiveType) {
    updateHasNull();
    if (primitiveType.getType().hasOffsets()) {
      // string type
      updateDataLen();
    }

    T t = doVisit(primitiveType);
    if (primitiveType.getType().hasOffsets()) {
      updateOffsets(
          true, // update offset buffer offset
          true,  // update data buffer offset
          false, // update slice info
          -1 // element size in bytes, not used for string
      );
    } else {
      updateOffsets(
          false, //update offset buffer offset
          true,  // update data buffer offset
          false,  // update slice info
          primitiveType.getType().getSizeInBytes() // element size in bytes
      );
    }
    currentIdx += 1;
    return t;
  }

  protected abstract T doVisit(Schema primitiveType);

  private void updateHasNull() {
    hasNull = false;
    for (KudoTable table : tables) {
      if (table.getHeader().hasValidityBuffer(currentIdx)) {
        hasNull = true;
        return;
      }
    }
  }

  // For string column only
  private void updateDataLen() {
    totalStrDataLen = 0;
    // String's data len needs to be calculated from offset buffer
    for (int tableIdx = 0; tableIdx < getTableSize(); tableIdx += 1) {
      SliceInfo sliceInfo = sliceInfoOf(tableIdx);
      if (sliceInfo.getRowCount() > 0) {
        int offset = offsetOf(tableIdx, 0);
        int endOffset = offsetOf(tableIdx, sliceInfo.getRowCount());

        strDataLen[tableIdx] = endOffset - offset;
        totalStrDataLen += strDataLen[tableIdx];
      } else {
        strDataLen[tableIdx] = 0;
      }
    }
  }

  private void updateOffsets(boolean hasOffset, boolean hasData, boolean updateSliceInfo, int sizeInBytes) {
    long totalRowCount = 0;
    for (int tableIdx = 0; tableIdx < tables.size(); tableIdx++) {
      SliceInfo sliceInfo = sliceInfoOf(tableIdx);
      if (sliceInfo.getRowCount() > 0) {
        if (updateSliceInfo) {
          int startOffset = offsetOf(tableIdx, 0);
          int endOffset = offsetOf(tableIdx, sliceInfo.getRowCount());
          int rowCount = endOffset - startOffset;
          totalRowCount += rowCount;

          sliceInfoStack[tableIdx].addLast(new SliceInfo(startOffset, rowCount));
        }

        long columnLen = 0;

        if (tables.get(tableIdx).getHeader().hasValidityBuffer(currentIdx)) {
          columnLen += padForHostAlignment(sliceInfo.getValidityBufferInfo().getBufferLength());
        }

        if (hasOffset) {
          columnLen += padForHostAlignment((sliceInfo.getRowCount() + 1) * Integer.BYTES);
          if (hasData) {
            // string type
            columnLen += padForHostAlignment(strDataLen[tableIdx]);
          }
          // otherwise list type
        } else {
          if (hasData) {
            // primitive type
            columnLen += padForHostAlignment(sliceInfo.getRowCount() * sizeInBytes);
          }
        }
        currentColumnOffsets[tableIdx] += columnLen;
      } else {
        if (updateSliceInfo) {
          sliceInfoStack[tableIdx].addLast(new SliceInfo(0, 0));
        }
      }
    }

    if (updateSliceInfo) {
      totalRowCountStack.addLast(toIntExact(totalRowCount));
    }

    System.out.println("After updateOffsets, current idx: " + currentIdx +
        ", totalRowCountStack: " + totalRowCountStack +
        ", currentColumnOffsets: " + Arrays.toString(currentColumnOffsets));
  }

  // Below parts are information about current column

  protected int getTotalRowCount() {
    return totalRowCountStack.getLast();
  }

  protected boolean hasNull() {
    return hasNull;
  }

  protected SliceInfo sliceInfoOf(int tableIdx) {
    return sliceInfoStack[tableIdx].getLast();
  }

  protected HostMemoryBuffer memoryBufferOf(int tableIdx) {
    return tables.get(tableIdx).getBuffer();
  }

  protected int offsetOf(int tableIdx, long rowIdx) {
    long startOffset;
    if (tables.get(tableIdx).getHeader().hasValidityBuffer(currentIdx)) {
      startOffset = currentColumnOffsets[tableIdx] + sliceInfoOf(tableIdx)
          .getValidityBufferInfo()
          .getBufferLength();
    } else {
      startOffset = currentColumnOffsets[tableIdx];
    }
    return tables.get(tableIdx).getBuffer().getInt(startOffset + rowIdx * Integer.BYTES);
  }

  protected long validifyBufferOffset(int tableIdx) {
    if (tables.get(tableIdx).getHeader().hasValidityBuffer(currentIdx)) {
      return currentColumnOffsets[tableIdx];
    } else {
      return INVALID_OFFSET;
    }
  }

  protected void copyDataBuffer(Schema type, HostMemoryBuffer dst, long dstOffset, int tableIdx, int dataLen) {
    long startOffset = currentColumnOffsets[tableIdx];
    if (tables.get(tableIdx).getHeader().hasValidityBuffer(currentIdx)) {
      startOffset += sliceInfoOf(tableIdx)
          .getValidityBufferInfo()
          .getBufferLength();
    }

    if (type.getType().hasOffsets()) {
      startOffset += (sliceInfoOf(tableIdx).getRowCount() + 1) * Integer.BYTES;
    }

    dst.copyFromHostBuffer(dstOffset, tables.get(tableIdx).getBuffer(), startOffset, dataLen);
  }

  protected long getTotalStrDataLen() {
    return totalStrDataLen;
  }

  protected int getStrDataLenOf(int tableIdx) {
    return strDataLen[tableIdx];
  }

  protected int getCurrentIdx() {
    return currentIdx;
  }

  public int getTableSize() {
    return this.tables.size();
  }
}
