package com.nvidia.spark.rapids.jni.kudo;

/**
 * This class contains metrics for serializing table using kudo format.
 */
public class WriteMetrics {
  private long calcHeaderTime;
  private long copyHeaderTime;
  private long copyBufferTime;
  private long writtenBytes;


  public WriteMetrics() {
    this.calcHeaderTime = 0;
    this.copyHeaderTime = 0;
    this.copyBufferTime = 0;
    this.writtenBytes = 0;
  }

  /**
   * Get the time spent on calculating the header.
   */
  public long getCalcHeaderTime() {
    return calcHeaderTime;
  }

  /**
   * Get the time spent on copying the buffer.
   */
  public long getCopyBufferTime() {
    return copyBufferTime;
  }

  public void addCopyBufferTime(long time) {
    copyBufferTime += time;
  }

  /**
   * Get the time spent on copying the header.
   */
  public long getCopyHeaderTime() {
    return copyHeaderTime;
  }

  public void addCalcHeaderTime(long time) {
    calcHeaderTime += time;
  }

  public void addCopyHeaderTime(long time) {
    copyHeaderTime += time;
  }

  /**
   * Get the number of bytes written.
   */
  public long getWrittenBytes() {
    return writtenBytes;
  }

  public void addWrittenBytes(long bytes) {
    writtenBytes += bytes;
  }
}
