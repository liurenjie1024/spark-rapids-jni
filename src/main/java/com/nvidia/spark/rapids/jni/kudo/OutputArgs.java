package com.nvidia.spark.rapids.jni.kudo;

import static com.nvidia.spark.rapids.jni.Preconditions.ensure;
import static com.nvidia.spark.rapids.jni.kudo.KudoSerializer.writerFrom;
import static java.util.Objects.requireNonNull;

import java.io.OutputStream;

public class OutputArgs {
  private final int rowOffset;
  private final int numRows;

  private final DataWriter dataWriter;
  private final SliceInfo sliceInfo;

  public OutputArgs(int rowOffset, int numRows, OutputStream outputStream) {
    ensure(rowOffset >= 0, () -> "rowOffset must be >= 0, but is " + rowOffset);
    ensure(numRows > 0, () -> "numRows must be >= 0, but is " + numRows);
    requireNonNull(outputStream, "outputStream is null");
    this.rowOffset = rowOffset;
    this.numRows = numRows;
    this.dataWriter =  writerFrom(outputStream); ;
    this.sliceInfo = new SliceInfo(rowOffset, numRows);
  }

  public int getRowOffset() {
    return rowOffset;
  }

  public int getNumRows() {
    return numRows;
  }

  public DataWriter getDataWriter() {
    return dataWriter;
  }

  public SliceInfo getSliceInfo() {
    return sliceInfo;
  }
}
