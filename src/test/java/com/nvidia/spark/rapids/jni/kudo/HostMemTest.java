package com.nvidia.spark.rapids.jni.kudo;

import static org.junit.jupiter.api.Assertions.assertEquals;

import ai.rapids.cudf.HostMemoryBuffer;
import org.junit.jupiter.api.Test;

public class HostMemTest {
  @Test
  public void getIntTest() {
    try(HostMemoryBuffer mem = HostMemoryBuffer.allocate(8)) {
      mem.setInt(0, 42);
      mem.setInt(4, 43);

      assertEquals(10, mem.getInt(1));
    }
  }
}
