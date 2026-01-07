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

import ai.rapids.cudf.HostMemoryBuffer;
import com.nvidia.spark.rapids.jni.Arms;

import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Optional;
import java.util.zip.CRC32;

import static com.nvidia.spark.rapids.jni.kudo.KudoSerializer.readerFrom;
import static java.util.Objects.requireNonNull;

/**
 * Serialized table in kudo format, including a {{@link KudoTableHeader}} and a {@link HostMemoryBuffer} for serialized
 * data.
 */
public class KudoTable implements AutoCloseable {
  private final KudoTableHeader header;
  private final HostMemoryBuffer buffer;

  /**
   * Create a kudo table.
   *
   * @param header kudo table header
   * @param buffer host memory buffer for the table data. KudoTable will take ownership of this buffer, so don't close
   *               it after passing it to this constructor.
   */
  public KudoTable(KudoTableHeader header, HostMemoryBuffer buffer) {
    requireNonNull(header, "Header must not be null");
    this.header = header;
    this.buffer = buffer;
  }

  /**
   * Read a kudo table from an input stream.
   *
   * @param in input stream
   * @return the kudo table, or empty if the input stream is empty.
   * @throws IOException if an I/O error occurs
   */
  public static Optional<KudoTable> from(InputStream in) throws IOException {
    requireNonNull(in, "Input stream must not be null");

    DataInputStream din = readerFrom(in);
    return KudoTableHeader.readFrom(din).map(header -> {
      // Header only
      if (header.getNumColumns() == 0) {
        return new KudoTable(header, null);
      }

      return Arms.closeIfException(HostMemoryBuffer.allocate(header.getTotalDataLen(), false), buffer -> {
        try {
          buffer.copyFromStream(0, din, header.getTotalDataLen());
          return new KudoTable(header, buffer);
        } catch (IOException e) {
          throw new RuntimeException(e);
        }
      });
    });
  }

  public KudoTableHeader getHeader() {
    return header;
  }

  public HostMemoryBuffer getBuffer() {
    return buffer;
  }

  @Override
  public String toString() {
    return "SerializedTable{" +
        "header=" + header +
        ", buffer=" + buffer +
        '}';
  }

  /**
   * Verify the CRC32 checksum of the data buffer.
   * The checksum is stored in the last 4 bytes of the buffer in big-endian format.
   *
   * @return true if the checksum is valid, false otherwise
   * @throws IllegalStateException if the buffer is null or too small to contain a checksum
   */
  public boolean verifyChecksum() {
    if (buffer == null) {
      throw new IllegalStateException("Cannot verify checksum on null buffer");
    }

    long bufferLen = buffer.getLength();
    if (bufferLen < Integer.BYTES) {
      throw new IllegalStateException("Buffer is too small to contain a checksum");
    }

    // The last 4 bytes contain the stored checksum in big-endian format
    long checksumOffset = bufferLen - Integer.BYTES;
    byte[] checksumBytes = new byte[4];
    buffer.getBytes(checksumBytes, 0, checksumOffset, 4);
    int storedChecksum = ((checksumBytes[0] & 0xFF) << 24) |
                        ((checksumBytes[1] & 0xFF) << 16) |
                        ((checksumBytes[2] & 0xFF) << 8) |
                        (checksumBytes[3] & 0xFF);

    // Calculate checksum over all data except the last 4 bytes (the checksum itself)
    CRC32 crc32 = new CRC32();
    long dataLen = bufferLen - Integer.BYTES;
    byte[] data = new byte[(int) dataLen];
    buffer.getBytes(data, 0, 0, (int) dataLen);
    crc32.update(data);

    int calculatedChecksum = (int) crc32.getValue();

    return storedChecksum == calculatedChecksum;
  }

  @Override
  public void close() throws Exception {
    if (buffer != null) {
      buffer.close();
    }
  }
}
