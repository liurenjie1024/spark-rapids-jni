package com.nvidia.spark.rapids.jni.kudo;

import ai.rapids.cudf.BufferType;
import ai.rapids.cudf.HostMemoryBuffer;
import ai.rapids.cudf.Schema;
import com.nvidia.spark.rapids.jni.Arms;
import com.nvidia.spark.rapids.jni.schema.SchemaVisitor;
import com.nvidia.spark.rapids.jni.schema.Visitors;

import java.util.Arrays;
import java.util.List;

import static com.nvidia.spark.rapids.jni.kudo.KudoSerializer.padForHostAlignment;
import static java.lang.Math.toIntExact;

abstract class BaseSlicedBufferMerger implements SchemaVisitor<Void, Void, Void> {
    private final KudoTable kudoTable;
    private final int[] destStartRows;
    private final List<ColumnOffsetInfo> columnOffsetInfoList;
    private final HostMemoryBuffer outputBuffer;
    private int curColumnIdx = 0;
    private int offset;

    BaseSlicedBufferMerger(KudoTable kudoTable, int[] destStartRows, BufferType bufferType,
                           List<ColumnOffsetInfo> columnOffsetInfoList, HostMemoryBuffer outputBuffer) {
        this.kudoTable = kudoTable;
        this.destStartRows = destStartRows;
        this.columnOffsetInfoList = columnOffsetInfoList;
        this.outputBuffer = outputBuffer;
        this.offset = kudoTable.getHeader().startOffsetOf(bufferType);
    }

    protected HostMemoryBuffer getOutputBuffer() {
        return outputBuffer;
    }

    protected KudoTable getKudoTable() {
        return kudoTable;
    }


    protected int getCurColumnIdx() {
        return curColumnIdx;
    }

    protected int getCurrentDestStartRows() {
        return destStartRows[curColumnIdx];
    }

    protected ColumnOffsetInfo getCurrentColumnOffsetInfo() {
        return columnOffsetInfoList.get(curColumnIdx);
    }

    protected int getOffset() {
        return offset;
    }

    protected void increaseOffset(int delta) {
        offset += toIntExact(padForHostAlignment(delta));
    }

    @Override
    public Void visitTopSchema(Schema schema, List<Void> children) {
        return null;
    }

    @Override
    public Void visitStruct(Schema structType, List<Void> children) {
        doVisitStruct();
        curColumnIdx += 1;
        return null;
    }

    abstract void doVisitStruct();

    @Override
    public Void preVisitList(Schema listType) {
        doPreVisitList();
        curColumnIdx += 1;
        return null;
    }

    abstract void doPreVisitList();

    @Override
    public Void visitList(Schema listType, Void preVisitResult, Void childResult) {
        doVisitList();
        return null;
    }

    abstract void doVisitList();

    @Override
    public Void visit(Schema primitiveType) {
        doVisitPrimitive(primitiveType);
        curColumnIdx += 1;
        return null;
    }

    abstract void doVisitPrimitive(Schema primitiveType);

    static KudoHostMergeResult merge(Schema schema, MergedInfoCalc mergedInfo) {
        return Arms.closeIfException(HostMemoryBuffer.allocate(mergedInfo.getTotalDataLen()),
                buffer -> {
                    // Flatten column count of schema
                    int colNum = mergedInfo.getColumnOffsets().size();
                    int[] rowCounts = new int[colNum];
                    int[] nullCounts = new int[colNum];
                    int[] destDataOffsets = new int[colNum];

                    for (KudoTable kudoTable : mergedInfo.getTables()) {
                        SlicedOffsetBufferMerger offsetBufferMerger = new SlicedOffsetBufferMerger(kudoTable,
                                rowCounts, mergedInfo.getColumnOffsets(), buffer);
                        Visitors.visitSchema(schema, offsetBufferMerger);


                        SlicedValidityBufferMerger validityBufferMerger = new SlicedValidityBufferMerger(kudoTable,
                                rowCounts, mergedInfo.getColumnOffsets(), buffer,
                                offsetBufferMerger.getOutputSliceInfos(), nullCounts);
                        Visitors.visitSchema(schema, validityBufferMerger);

                        SlicedDataBufferMerger dataBufferMerger = new SlicedDataBufferMerger(kudoTable, rowCounts,
                                mergedInfo.getColumnOffsets(), buffer, offsetBufferMerger.getDataLen(), destDataOffsets);
                        Visitors.visitSchema(schema, dataBufferMerger);

                        for (int i = 0; i < colNum; i++) {
                            rowCounts[i] += offsetBufferMerger.getOutputSliceInfos()[i].getRowCount();
                        }
                    }

                    HostMergeResultCalc calc = new HostMergeResultCalc(mergedInfo.getColumnOffsets(), nullCounts, rowCounts, buffer);
                    KudoHostMergeResult result = Visitors.visitSchema(schema, calc);
                    System.out.println("Merged result: " + result);
                    return result;
                });
    }
}
