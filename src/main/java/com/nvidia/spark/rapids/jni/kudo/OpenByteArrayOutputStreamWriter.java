package com.nvidia.spark.rapids.jni.kudo;

import static java.util.Objects.requireNonNull;

import ai.rapids.cudf.HostMemoryBuffer;
import java.io.IOException;

public class OpenByteArrayOutputStreamWriter implements DataWriter {
  private final OpenByteArrayOutputStream out;

  public OpenByteArrayOutputStreamWriter(OpenByteArrayOutputStream bout) {
    requireNonNull(bout, "Byte array output stream can't be null");
    this.out = bout;
  }

  @Override
  public void reserve(int size) throws IOException {
    out.ensureCapacity(size);
  }

  @Override
  public void writeInt(int v) throws IOException {
    out.ensureCapacity(4 + out.size());
    out.write((v >>> 24) & 0xFF);
    out.write((v >>> 16) & 0xFF);
    out.write((v >>>  8) & 0xFF);
    out.write((v >>>  0) & 0xFF);
  }

  @Override
  public void copyDataFrom(HostMemoryBuffer src, long srcOffset, long len) throws IOException {
    out.write(src, srcOffset, len);
  }

  @Override
  public void flush() throws IOException {

  }

  @Override
  public void write(byte[] arr, int offset, int length) throws IOException {
    out.write(arr, offset, length);
  }
}
