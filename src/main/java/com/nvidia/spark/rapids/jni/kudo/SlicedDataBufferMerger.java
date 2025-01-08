package com.nvidia.spark.rapids.jni.kudo;

import ai.rapids.cudf.BufferType;
import ai.rapids.cudf.HostMemoryBuffer;
import ai.rapids.cudf.Schema;

import java.util.List;

class SlicedDataBufferMerger extends BaseSlicedBufferMerger {
    private final int[] inputDataLen;
    private final int[] outputDataOffset;

    SlicedDataBufferMerger(KudoTable kudoTable, int[] destStartRows, List<ColumnOffsetInfo> columnOffsetInfoList,
                           HostMemoryBuffer outputBuffer,
                           int[] inputDataLen, int[] outputDataOffset) {
        super(kudoTable, destStartRows, BufferType.DATA, columnOffsetInfoList, outputBuffer);
        this.inputDataLen = inputDataLen;
        this.outputDataOffset = outputDataOffset;
    }

    @Override
    void doVisitStruct() {
    }

    @Override
    void doPreVisitList() {
    }

    @Override
    void doVisitList() {
    }

    @Override
    void doVisitPrimitive(Schema primitiveType) {
        deserializeDataBuffer();
    }

    private void deserializeDataBuffer() {
        ColumnOffsetInfo columnOffsetInfo = getCurrentColumnOffsetInfo();
        if (columnOffsetInfo.getData() == ColumnOffsetInfo.INVALID_OFFSET || columnOffsetInfo.getDataBufferLen() == 0) {
            return;
        }

        System.out.println("Data buffer column offset: " + columnOffsetInfo);
        int curColIdx = getCurColumnIdx();
        getOutputBuffer().copyFromHostBuffer(columnOffsetInfo.getData() + outputDataOffset[curColIdx],
                getKudoTable().getBuffer(), getOffset(), inputDataLen[curColIdx]);

        increaseOffset(inputDataLen[curColIdx]);
        outputDataOffset[curColIdx] += inputDataLen[curColIdx];
    }
}
