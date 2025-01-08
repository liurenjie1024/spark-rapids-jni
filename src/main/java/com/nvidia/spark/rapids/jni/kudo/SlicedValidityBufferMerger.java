package com.nvidia.spark.rapids.jni.kudo;

import ai.rapids.cudf.BufferType;
import ai.rapids.cudf.HostMemoryBuffer;
import ai.rapids.cudf.Schema;

import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.List;

import static java.lang.Math.min;
import static java.lang.Math.toIntExact;

class SlicedValidityBufferMerger extends BaseSlicedBufferMerger {
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

    private final SliceInfo[] sliceInfoList;
    private final int[] nullCount;

    SlicedValidityBufferMerger(KudoTable kudoTable, int[] destStartRows, List<ColumnOffsetInfo> columnOffsetInfoList,
                               HostMemoryBuffer outputBuffer, SliceInfo[] sliceInfoList, int[] nullCount) {
        super(kudoTable, destStartRows, BufferType.VALIDITY, columnOffsetInfoList, outputBuffer);
        this.sliceInfoList = sliceInfoList;
        this.nullCount = nullCount;
    }

    @Override
    void doVisitStruct() {
        deserializeValidityBuffer();
    }

    @Override
    void doPreVisitList() {
        deserializeValidityBuffer();
    }

    @Override
    void doVisitList() {
    }

    @Override
    void doVisitPrimitive(Schema primitiveType) {
        deserializeValidityBuffer();
    }

    private void deserializeValidityBuffer() {
        ColumnOffsetInfo columnOffsetInfo = getCurrentColumnOffsetInfo();
        if (columnOffsetInfo.getValidity() == ColumnOffsetInfo.INVALID_OFFSET) {
            return;
        }

        int curColIdx = getCurColumnIdx();

        SliceInfo sliceInfo = sliceInfoList[curColIdx];

        if (getKudoTable().getHeader().hasValidityBuffer(curColIdx)) {
            nullCount[curColIdx] += copyValidityBuffer(getOutputBuffer()
                            .asByteBuffer(columnOffsetInfo.getValidity(),
                            toIntExact(columnOffsetInfo.getValidityBufferLen())),
                    getCurrentDestStartRows(), getKudoTable().getBuffer().asByteBuffer(getOffset(),
                            sliceInfo.getValidityBufferInfo().getBufferLength()), sliceInfo);
            increaseOffset(sliceInfo.getValidityBufferInfo().getBufferLength());
        } else {
            appendAllValid(getOutputBuffer(), columnOffsetInfo.getValidity(), getCurrentDestStartRows(),
                    sliceInfo.getRowCount());
        }
    }

    /**
     * Copy a sliced validity buffer to the destination buffer, starting at the given bit offset.
     *
     * @return Number of nulls in the validity buffer.
     */
    private static int copyValidityBuffer(ByteBuffer destBuffer,
                                          int destStartRow,
                                          ByteBuffer srcBuffer,
                                          SliceInfo sliceInfo) {
        int nullCount = 0;
        int totalRowCount = sliceInfo.getRowCount();
        int curIdx = 0;
        int curSrcByteIdx = 0;
        int curSrcBitIdx = sliceInfo.getValidityBufferInfo().getBeginBit();
        int curDestByteIdx = destStartRow / 8;
        int curDestBitIdx = destStartRow % 8;

        while (curIdx < totalRowCount) {
            int leftRowCount = totalRowCount - curIdx;
            int appendCount;
            if (curDestBitIdx == 0) {
                appendCount = min(8, leftRowCount);
            } else {
                appendCount = min(8 - curDestBitIdx, leftRowCount);
            }

            int leftBitsInCurSrcByte = 8 - curSrcBitIdx;
            byte srcByte = srcBuffer.get(curSrcByteIdx);
            if (leftBitsInCurSrcByte >= appendCount) {
                // Extract appendCount bits from srcByte, starting from curSrcBitIdx
                byte mask = (byte) (((1 << appendCount) - 1) & 0xFF);
                srcByte = (byte) ((srcByte >>> curSrcBitIdx) & mask);

                nullCount += (appendCount - NUMBER_OF_ONES[srcByte & 0xFF]);

                // Sets the bits in destination buffer starting from curDestBitIdx to 0
                byte destByte = destBuffer.get(curDestByteIdx);
                destByte = (byte) (destByte & ((1 << curDestBitIdx) - 1) & 0xFF);

                // Update destination byte with the bits from source byte
                destByte = (byte) ((destByte | (srcByte << curDestBitIdx)) & 0xFF);
                destBuffer.put(curDestByteIdx, destByte);

                curSrcBitIdx += appendCount;
                if (curSrcBitIdx == 8) {
                    curSrcBitIdx = 0;
                    curSrcByteIdx += 1;
                }
            } else {
                // Extract appendCount bits from srcByte, starting from curSrcBitIdx
                byte mask = (byte) (((1 << leftBitsInCurSrcByte) - 1) & 0xFF);
                srcByte = (byte) ((srcByte >>> curSrcBitIdx) & mask);

                byte nextSrcByte = srcBuffer.get(curSrcByteIdx + 1);
                byte nextSrcByteMask = (byte) ((1 << (appendCount - leftBitsInCurSrcByte)) - 1);
                nextSrcByte = (byte) (nextSrcByte & nextSrcByteMask);
                nextSrcByte = (byte) (nextSrcByte << leftBitsInCurSrcByte);
                srcByte = (byte) (srcByte | nextSrcByte);

                nullCount += (appendCount - NUMBER_OF_ONES[srcByte & 0xFF]);

                // Sets the bits in destination buffer starting from curDestBitIdx to 0
                byte destByte = destBuffer.get(curDestByteIdx);
                destByte = (byte) (destByte & ((1 << curDestBitIdx) - 1));

                // Update destination byte with the bits from source byte
                destByte = (byte) (destByte | (srcByte << curDestBitIdx));
                destBuffer.put(curDestByteIdx, destByte);

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

    private static void appendAllValid(HostMemoryBuffer dest, long offset, int startBit, int numRowsLong) {
        int numRows = toIntExact(numRowsLong);
        int curDestByteIdx = toIntExact(offset + startBit / 8);
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
}
