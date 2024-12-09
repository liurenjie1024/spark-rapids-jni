package com.nvidia.spark.rapids.jni.kudo;

import static java.lang.Math.toIntExact;
import static java.util.Objects.requireNonNull;

import ai.rapids.cudf.HostMemoryBuffer;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;

public class ByteArrayOutputStreamWriter implements DataWriter {
  private static final Method ENSURE_CAPACITY;
  private static final Field BUF_FIELD;
  private static final Field COUNT_FIELD;

  static {
    try {
      ENSURE_CAPACITY = ByteArrayOutputStream.class.getDeclaredMethod("ensureCapacity", int.class);
      ENSURE_CAPACITY.setAccessible(true);

      BUF_FIELD = ByteArrayOutputStream.class.getDeclaredField("buf");
      BUF_FIELD.setAccessible(true);

      COUNT_FIELD = ByteArrayOutputStream.class.getDeclaredField("count");
      COUNT_FIELD.setAccessible(true);
    } catch (NoSuchMethodException e) {
      throw new RuntimeException("Failed to find ensureCapacity method", e);
    } catch (NoSuchFieldException e) {
      throw new RuntimeException(e);
    }
  }



  private final ByteArrayOutputStream out;

  public ByteArrayOutputStreamWriter(ByteArrayOutputStream bout) {
    requireNonNull(bout, "Byte array output stream can't be null");
    this.out = bout;
  }

  @Override
  public void writeInt(int v) throws IOException {
    out.write((v >>> 24) & 0xFF);
    out.write((v >>> 16) & 0xFF);
    out.write((v >>>  8) & 0xFF);
    out.write((v >>>  0) & 0xFF);
  }

  @Override
  public void copyDataFrom(HostMemoryBuffer src, long srcOffset, long len) throws IOException {
    try {
      int lenInt = toIntExact(len);
      int count = (int) COUNT_FIELD.get(out);

      ENSURE_CAPACITY.invoke(out, len + count);
      byte[] buf = (byte[]) BUF_FIELD.get(out);


      src.getBytes(buf, count, srcOffset, lenInt);
      COUNT_FIELD.set(out, count + lenInt);
    } catch (IllegalAccessException e) {
      throw new RuntimeException(e);
    } catch (InvocationTargetException e) {
      throw new RuntimeException(e);
    }
  }

  @Override
  public void flush() throws IOException {

  }

  @Override
  public void write(byte[] arr, int offset, int length) throws IOException {
    out.write(arr, offset, length);
  }
}
