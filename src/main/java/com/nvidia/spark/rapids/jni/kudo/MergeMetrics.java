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

public class MergeMetrics {
  // The time it took to calculate combined header in nanoseconds
  private final long calcHeaderTime;
  // The time it took to merge the buffers into the host buffer in nanoseconds
  private final long mergeIntoHostBufferTime;
  // The time it took to convert the host buffer into a contiguous table in nanoseconds
  private final long convertIntoContiguousTableTime;

  public MergeMetrics(long calcHeaderTime, long mergeIntoHostBufferTime,
      long convertIntoContiguousTableTime) {
    this.calcHeaderTime = calcHeaderTime;
    this.mergeIntoHostBufferTime = mergeIntoHostBufferTime;
    this.convertIntoContiguousTableTime = convertIntoContiguousTableTime;
  }

  public long getCalcHeaderTime() {
    return calcHeaderTime;
  }

  public long getMergeIntoHostBufferTime() {
    return mergeIntoHostBufferTime;
  }

  public long getConvertIntoContiguousTableTime() {
    return convertIntoContiguousTableTime;
  }

  public static Builder builder() {
    return new Builder();
  }

  public static Builder builder(MergeMetrics metrics) {
    return new Builder()
        .calcHeaderTime(metrics.calcHeaderTime)
        .mergeIntoHostBufferTime(metrics.mergeIntoHostBufferTime)
        .convertIntoContiguousTableTime(metrics.convertIntoContiguousTableTime);
  }


  public static class Builder {
    private long calcHeaderTime;
    private long mergeIntoHostBufferTime;
    private long convertIntoContiguousTableTime;

    public Builder calcHeaderTime(long calcHeaderTime) {
      this.calcHeaderTime = calcHeaderTime;
      return this;
    }

    public Builder mergeIntoHostBufferTime(long mergeIntoHostBufferTime) {
      this.mergeIntoHostBufferTime = mergeIntoHostBufferTime;
      return this;
    }

    public Builder convertIntoContiguousTableTime(long convertIntoContiguousTableTime) {
      this.convertIntoContiguousTableTime = convertIntoContiguousTableTime;
      return this;
    }

    public MergeMetrics build() {
      return new MergeMetrics(calcHeaderTime, mergeIntoHostBufferTime, convertIntoContiguousTableTime);
    }
  }
}
