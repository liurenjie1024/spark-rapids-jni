package com.nvidia.spark.rapids.jni.kudo;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import ai.rapids.cudf.HostMemoryBuffer;
import java.util.Arrays;
import org.junit.jupiter.api.Test;

public class HostMemTest {
  @Test
  public void getIntTest() {
    int[] ints = new int[100];
    for (int i = 0; i < 100; i++) {
      ints[i] = i * 10;
    }
    try(HostMemoryBuffer mem1 = HostMemoryBuffer.allocate(4 * 100)) {
      mem1.setInts(0, ints, 0, 100);
      try(HostMemoryBuffer mem2 = HostMemoryBuffer.allocate(8* 100)) {
        mem2.copyFromHostBuffer(37, mem1, 0, 4 * 100);
        int[] ints2 = new int[100];
        mem2.getInts(ints2, 0, 37, 100);
        assertArrayEquals(ints, ints2);
      }
    }
  }
}
