package com.nvidia.spark.rapids.jni.kudo;

import ai.rapids.cudf.BufferType;
import ai.rapids.cudf.HostMemoryBuffer;
import ai.rapids.cudf.Schema;

import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.List;

import static java.lang.Math.toIntExact;

class SlicedOffsetBufferMerger extends BaseSlicedBufferMerger {
    private final SliceInfo[] outputSliceInfos;
    private final int[] dataLen;
    private final Deque<SliceInfo> sliceInfoStack = new ArrayDeque<>(16);

    SlicedOffsetBufferMerger(KudoTable kudoTable, int[] destStartRows, List<ColumnOffsetInfo> columnOffsetInfoList,
                             HostMemoryBuffer outputBuffer) {
        super(kudoTable, destStartRows, BufferType.OFFSET, columnOffsetInfoList, outputBuffer);
        outputSliceInfos = new SliceInfo[destStartRows.length];
        dataLen = new int[destStartRows.length];
        sliceInfoStack.addLast(new SliceInfo(kudoTable.getHeader().getOffset(), kudoTable.getHeader().getNumRows()));
    }

    public SliceInfo[] getOutputSliceInfos() {
        return outputSliceInfos;
    }

    public int[] getDataLen() {
        return dataLen;
    }

    @Override
    void doVisitStruct() {
        int curColumnIdx = getCurColumnIdx();
        outputSliceInfos[curColumnIdx] = sliceInfoStack.getLast();
        dataLen[curColumnIdx] = 0;
        deserializeOffset();
    }

    @Override
    void doPreVisitList() {
        int curColumnIdx = getCurColumnIdx();
        outputSliceInfos[curColumnIdx] = sliceInfoStack.getLast();
        dataLen[curColumnIdx] = 0;
        sliceInfoStack.addLast(deserializeOffset());
    }

    @Override
    void doVisitList() {
        sliceInfoStack.removeLast();
    }

    @Override
    void doVisitPrimitive(Schema primitiveType) {
        int curColumnIdx = getCurColumnIdx();
        outputSliceInfos[curColumnIdx] = sliceInfoStack.getLast();
        SliceInfo sliceInfo = deserializeOffset();
        if (primitiveType.getType().hasOffsets()) {
            // String type
            dataLen[curColumnIdx] = sliceInfo.getRowCount();
        } else {
            // Fix sized type
            dataLen[curColumnIdx] = outputSliceInfos[curColumnIdx].getRowCount() *
                    primitiveType.getType().getSizeInBytes();
        }
    }

    private SliceInfo deserializeOffset() {
        SliceInfo sliceInfo = sliceInfoStack.getLast();
        ColumnOffsetInfo columnOffsetInfo = getCurrentColumnOffsetInfo();
        if (columnOffsetInfo.getOffset() == ColumnOffsetInfo.INVALID_OFFSET ||
                columnOffsetInfo.getOffsetBufferLen() == 0) {
            return sliceInfo;
        }

        if (sliceInfo.getRowCount() <= 0) {
            return sliceInfo;
        }

        int bufferSize = toIntExact(Integer.BYTES * (sliceInfo.getRowCount() + 1));

        IntBuffer inputOffsetBuffer = getKudoTable().getBuffer()
                .asByteBuffer(getOffset(), bufferSize)
                .order(ByteOrder.LITTLE_ENDIAN)
                .asIntBuffer();

        IntBuffer outputOffsetBuffer = getOutputBuffer()
                .asByteBuffer(columnOffsetInfo.getOffset() + getCurrentDestStartRows() * Integer.BYTES, bufferSize)
                .order(ByteOrder.LITTLE_ENDIAN)
                .asIntBuffer();

        System.out.println("Slice offset deserialize, source offset: " + getOffset()
                + ", buffer size: " + bufferSize
                + ", dest offset: " + columnOffsetInfo.getOffset()
                + ", dest start row count: " + getCurrentDestStartRows());

        int startOffset = inputOffsetBuffer.get(0);
        int accumulatedOffset;
        if (getCurrentDestStartRows() == 0) {
            accumulatedOffset = 0;
        } else {
            accumulatedOffset = outputOffsetBuffer.get(0);
        }
        for (int i = 0; i <= sliceInfo.getRowCount(); i++) {
            outputOffsetBuffer.put(i, inputOffsetBuffer.get(i) - startOffset + accumulatedOffset);
        }

        increaseOffset(bufferSize);

        return new SliceInfo(startOffset, inputOffsetBuffer.get(sliceInfo.getRowCount()) - startOffset);
    }
}
