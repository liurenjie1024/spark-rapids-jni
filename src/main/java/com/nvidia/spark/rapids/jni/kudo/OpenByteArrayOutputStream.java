package com.nvidia.spark.rapids.jni.kudo;

import ai.rapids.cudf.HostMemoryBuffer;
import java.io.IOException;
import java.io.OutputStream;
import java.io.UnsupportedEncodingException;
import java.util.Arrays;

public class OpenByteArrayOutputStream extends OutputStream {

  protected byte buf[];

  /**
   * The number of valid bytes in the buffer.
   */
  protected int count;

  /**
   * Creates a new byte array output stream. The buffer capacity is
   * initially 32 bytes, though its size increases if necessary.
   */
  public OpenByteArrayOutputStream() {
    this(32);
  }

  /**
   * Creates a new byte array output stream, with a buffer capacity of
   * the specified size, in bytes.
   *
   * @param   size   the initial size.
   * @exception  IllegalArgumentException if size is negative.
   */
  public OpenByteArrayOutputStream(int size) {
    if (size < 0) {
      throw new IllegalArgumentException("Negative initial size: "
          + size);
    }
    buf = new byte[size];
  }

  void ensureCapacity(int minCapacity) {
    // overflow-conscious code
    if (minCapacity - buf.length > 0)
      grow(minCapacity);
  }

  /**
   * The maximum size of array to allocate.
   * Some VMs reserve some header words in an array.
   * Attempts to allocate larger arrays may result in
   * OutOfMemoryError: Requested array size exceeds VM limit
   */
  private static final int MAX_ARRAY_SIZE = Integer.MAX_VALUE - 8;

  /**
   * Increases the capacity to ensure that it can hold at least the
   * number of elements specified by the minimum capacity argument.
   *
   * @param minCapacity the desired minimum capacity
   */
  private void grow(int minCapacity) {
    // overflow-conscious code
    int oldCapacity = buf.length;
    int newCapacity = oldCapacity << 1;
    if (newCapacity - minCapacity < 0)
      newCapacity = minCapacity;
    if (newCapacity - MAX_ARRAY_SIZE > 0)
      newCapacity = hugeCapacity(minCapacity);
    buf = Arrays.copyOf(buf, newCapacity);
  }

  private static int hugeCapacity(int minCapacity) {
    if (minCapacity < 0) // overflow
      throw new OutOfMemoryError();
    return (minCapacity > MAX_ARRAY_SIZE) ?
        Integer.MAX_VALUE :
        MAX_ARRAY_SIZE;
  }

  /**
   * Writes the specified byte to this byte array output stream.
   *
   * @param   b   the byte to be written.
   */
  public void write(int b) {
    ensureCapacity(count + 1);
    buf[count] = (byte) b;
    count += 1;
  }

  /**
   * Writes <code>len</code> bytes from the specified byte array
   * starting at offset <code>off</code> to this byte array output stream.
   *
   * @param   b     the data.
   * @param   off   the start offset in the data.
   * @param   len   the number of bytes to write.
   */
  public void write(byte b[], int off, int len) {
    if ((off < 0) || (off > b.length) || (len < 0) ||
        ((off + len) - b.length > 0)) {
      throw new IndexOutOfBoundsException();
    }
    ensureCapacity(count + len);
    System.arraycopy(b, off, buf, count, len);
    count += len;
  }

  public void write(HostMemoryBuffer src, long srcOffset, long len) throws IOException {
    int lenInt = Math.toIntExact(len);
    ensureCapacity(count + lenInt);
    src.getBytes(buf, count, srcOffset, lenInt);
    count += lenInt;
  }

  public byte[] getBuf() {
    return buf;
  }

  /**
   * Writes the complete contents of this byte array output stream to
   * the specified output stream argument, as if by calling the output
   * stream's write method using <code>out.write(buf, 0, count)</code>.
   *
   * @param      out   the output stream to which to write the data.
   * @exception IOException  if an I/O error occurs.
   */
  public synchronized void writeTo(OutputStream out) throws IOException {
    out.write(buf, 0, count);
  }

  public synchronized void reset() {
    count = 0;
  }

  /**
   * Creates a newly allocated byte array. Its size is the current
   * size of this output stream and the valid contents of the buffer
   * have been copied into it.
   *
   * @return  the current contents of this output stream, as a byte array.
   * @see     java.io.ByteArrayOutputStream#size()
   */
  public synchronized byte toByteArray()[] {
    return Arrays.copyOf(buf, count);
  }

  public int size() {
    return count;
  }

  /**
   * Converts the buffer's contents into a string decoding bytes using the
   * platform's default character set. The length of the new <tt>String</tt>
   * is a function of the character set, and hence may not be equal to the
   * size of the buffer.
   *
   * <p> This method always replaces malformed-input and unmappable-character
   * sequences with the default replacement string for the platform's
   * default character set. The {@linkplain java.nio.charset.CharsetDecoder}
   * class should be used when more control over the decoding process is
   * required.
   *
   * @return String decoded from the buffer's contents.
   * @since  JDK1.1
   */
  public synchronized String toString() {
    return new String(buf, 0, count);
  }

  /**
   * Converts the buffer's contents into a string by decoding the bytes using
   * the named {@link java.nio.charset.Charset charset}. The length of the new
   * <tt>String</tt> is a function of the charset, and hence may not be equal
   * to the length of the byte array.
   *
   * <p> This method always replaces malformed-input and unmappable-character
   * sequences with this charset's default replacement string. The {@link
   * java.nio.charset.CharsetDecoder} class should be used when more control
   * over the decoding process is required.
   *
   * @param      charsetName  the name of a supported
   *             {@link java.nio.charset.Charset charset}
   * @return     String decoded from the buffer's contents.
   * @exception UnsupportedEncodingException
   *             If the named charset is not supported
   * @since      JDK1.1
   */
  public synchronized String toString(String charsetName)
      throws UnsupportedEncodingException
  {
    return new String(buf, 0, count, charsetName);
  }

  /**
   * Creates a newly allocated string. Its size is the current size of
   * the output stream and the valid contents of the buffer have been
   * copied into it. Each character <i>c</i> in the resulting string is
   * constructed from the corresponding element <i>b</i> in the byte
   * array such that:
   * <blockquote><pre>
   *     c == (char)(((hibyte &amp; 0xff) &lt;&lt; 8) | (b &amp; 0xff))
   * </pre></blockquote>
   *
   * @deprecated This method does not properly convert bytes into characters.
   * As of JDK&nbsp;1.1, the preferred way to do this is via the
   * <code>toString(String enc)</code> method, which takes an encoding-name
   * argument, or the <code>toString()</code> method, which uses the
   * platform's default character encoding.
   *
   * @param      hibyte    the high byte of each resulting Unicode character.
   * @return     the current contents of the output stream, as a string.
   * @see        java.io.ByteArrayOutputStream#size()
   * @see        java.io.ByteArrayOutputStream#toString(String)
   * @see        java.io.ByteArrayOutputStream#toString()
   */
  @Deprecated
  public synchronized String toString(int hibyte) {
    return new String(buf, hibyte, 0, count);
  }

  /**
   * Closing a <tt>ByteArrayOutputStream</tt> has no effect. The methods in
   * this class can be called after the stream has been closed without
   * generating an <tt>IOException</tt>.
   */
  public void close() throws IOException {
  }
}
