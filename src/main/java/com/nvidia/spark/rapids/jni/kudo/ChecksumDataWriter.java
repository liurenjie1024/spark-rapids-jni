/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

import ai.rapids.cudf.HostMemoryBuffer;

import java.io.IOException;
import java.util.zip.CRC32;

/**
 * A DataWriter wrapper that calculates CRC32 checksum of the data being written.
 * This is used to verify data integrity of serialized Kudo tables.
 */
class ChecksumDataWriter implements DataWriter {
  private final DataWriter underlying;
  private final CRC32 crc32;

  public ChecksumDataWriter(DataWriter underlying) {
    this.underlying = underlying;
    this.crc32 = new CRC32();
  }

  @Override
  public void writeInt(int i) throws IOException {
    underlying.writeInt(i);
    // Update checksum with the int value in big-endian format
    byte[] bytes = new byte[4];
    bytes[0] = (byte) ((i >>> 24) & 0xFF);
    bytes[1] = (byte) ((i >>> 16) & 0xFF);
    bytes[2] = (byte) ((i >>> 8) & 0xFF);
    bytes[3] = (byte) (i & 0xFF);
    crc32.update(bytes);
  }

  @Override
  public void reserve(int size) throws IOException {
    underlying.reserve(size);
  }

  @Override
  public void copyDataFrom(HostMemoryBuffer src, long srcOffset, long len) throws IOException {
    underlying.copyDataFrom(src, srcOffset, len);
    // Update checksum with the data from the buffer
    byte[] bytes = new byte[(int) len];
    src.getBytes(bytes, 0, srcOffset, (int) len);
    crc32.update(bytes);
  }

  @Override
  public void flush() throws IOException {
    underlying.flush();
  }

  @Override
  public void write(byte[] arr, int offset, int length) throws IOException {
    underlying.write(arr, offset, length);
    // Update checksum with the byte array
    crc32.update(arr, offset, length);
  }

  /**
   * Get the current CRC32 checksum value.
   *
   * @return the checksum value as a long
   */
  public long getChecksumValue() {
    return crc32.getValue();
  }

  /**
   * Write the checksum value to the underlying writer.
   * The checksum is written as a 4-byte integer in network byte order.
   *
   * @throws IOException if an I/O error occurs
   */
  public void writeChecksum() throws IOException {
    // Write checksum without updating the checksum itself
    underlying.writeInt((int) crc32.getValue());
  }
}

