package com.nvidia.spark.rapids.jni.kudo;

import ai.rapids.cudf.HostMemoryBuffer;
import ai.rapids.cudf.Schema;
import com.nvidia.spark.rapids.jni.schema.SchemaVisitor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

class HostMergeResultCalc implements SchemaVisitor<Void, Void, KudoHostMergeResult> {
    private final List<ColumnOffsetInfo> columnOffsetInfos;
    private final int[] nullCounts;
    private final int[] rowCounts;
    private final HostMemoryBuffer hostMemoryBuffer;

    private final ColumnViewInfo[] columnViewInfos;
    private int curColIdx = 0;

    HostMergeResultCalc(List<ColumnOffsetInfo> columnOffsetInfos, int[] nullCounts, int[] rowCounts,
                        HostMemoryBuffer hostMemoryBuffer) {
        this.columnOffsetInfos = columnOffsetInfos;
        this.nullCounts = nullCounts;
        this.rowCounts = rowCounts;
        this.columnViewInfos = new ColumnViewInfo[columnOffsetInfos.size()];
        this.hostMemoryBuffer = hostMemoryBuffer;
    }

    @Override
    public KudoHostMergeResult visitTopSchema(Schema schema, List<Void> children) {
        List<ColumnViewInfo> columnViewInfoList = new ArrayList<>(columnViewInfos.length);
        columnViewInfoList.addAll(Arrays.asList(columnViewInfos));
        return new KudoHostMergeResult(schema, hostMemoryBuffer, columnViewInfoList);
    }

    @Override
    public Void visitStruct(Schema structType, List<Void> children) {
        columnViewInfos[curColIdx] = new ColumnViewInfo(structType.getType(),
                columnOffsetInfos.get(curColIdx), nullCounts[curColIdx], rowCounts[curColIdx]);
        curColIdx++;
        return null;
    }

    @Override
    public Void preVisitList(Schema listType) {
        columnViewInfos[curColIdx] = new ColumnViewInfo(listType.getType(),
                columnOffsetInfos.get(curColIdx), nullCounts[curColIdx], rowCounts[curColIdx]);
        curColIdx++;
        return null;
    }

    @Override
    public Void visitList(Schema listType, Void preVisitResult, Void childResult) {
        return null;
    }

    @Override
    public Void visit(Schema primitiveType) {
        columnViewInfos[curColIdx] = new ColumnViewInfo(primitiveType.getType(),
                columnOffsetInfos.get(curColIdx), nullCounts[curColIdx], rowCounts[curColIdx]);
        curColIdx++;
        return null;
    }
}
