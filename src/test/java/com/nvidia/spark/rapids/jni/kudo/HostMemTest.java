package com.nvidia.spark.rapids.jni.kudo;

import static org.junit.jupiter.api.Assertions.assertEquals;

import ai.rapids.cudf.HostMemoryBuffer;
import java.util.Arrays;
import org.junit.jupiter.api.Test;

public class HostMemTest {
  @Test
  public void getIntTest() {
    try(HostMemoryBuffer mem = HostMemoryBuffer.allocate(1000)) {
      for (int i = 0; i < 1000 / 4; i++) {
        mem.setInt(i * 4, i * 10);
      }

      int[] ints = new int[100];
      mem.getInts(ints, 0, 37, 100);

      System.out.println("ints:" + Arrays.toString(ints));
    }
  }
}
