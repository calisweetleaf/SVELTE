import pytest
import os
import json
import struct
import shutil
from src.utils import file_io
from src.utils.file_io import FileIOException
from typing import Union # Added this import back

# --- Helper Functions for Tests ---

def create_dummy_file(filepath: str, content: Union[str, bytes], binary: bool = False):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    mode = 'wb' if binary else 'w'
    encoding = None if binary else 'utf-8'
    with open(filepath, mode, encoding=encoding) as f:
        f.write(content)

# --- Tests for each function in file_io.py ---

class TestFileIOUtils:

    # read_binary_file
    def test_read_binary_file_success(self, tmp_path):
        content = b"binary\x00data"
        file = tmp_path / "test.bin"
        create_dummy_file(str(file), content, binary=True)
        assert file_io.read_binary_file(str(file)) == content

    def test_read_binary_file_not_found(self, tmp_path):
        with pytest.raises(FileIOException, match="File not found"):
            file_io.read_binary_file(str(tmp_path / "nonexistent.bin"))

    # write_json and read_json
    def test_write_read_json_atomic(self, tmp_path):
        data = {"key": "value", "number": 123, "nested": {"pi": 3.14}}
        file = tmp_path / "test_atomic.json"

        file_io.write_json(data, str(file), atomic=True, sort_keys=True)
        assert file.exists()

        read_data = file_io.read_json(str(file))
        assert read_data == data

    def test_write_read_json_non_atomic(self, tmp_path):
        data = {"key": "value", "number": 123}
        file = tmp_path / "subdir" / "test_nonatomic.json" # Test subdirectory creation

        file_io.write_json(data, str(file), atomic=False)
        assert file.exists()

        read_data = file_io.read_json(str(file))
        assert read_data == data

    def test_read_json_not_found(self, tmp_path):
        with pytest.raises(FileIOException, match="File not found"):
            file_io.read_json(str(tmp_path / "nonexistent.json"))

    def test_read_json_invalid_json(self, tmp_path):
        file = tmp_path / "invalid.json"
        create_dummy_file(str(file), "this is not json")
        with pytest.raises(FileIOException, match="Failed to read JSON"):
            file_io.read_json(str(file))

    def test_read_json_object_hook(self, tmp_path):
        data_str = '{"custom_obj": true, "value": 10}'
        file = tmp_path / "hook.json"
        create_dummy_file(str(file), data_str)

        def custom_decoder(dct):
            if "custom_obj" in dct:
                return "HOOKED_" + str(dct["value"])
            return dct

        read_data = file_io.read_json(str(file), object_hook=custom_decoder)
        assert read_data == "HOOKED_10"


    # ensure_dir
    def test_ensure_dir_creates_new(self, tmp_path):
        dir_path = tmp_path / "new_dir" / "sub_dir"
        assert not dir_path.exists()
        file_io.ensure_dir(str(dir_path))
        assert dir_path.exists()
        assert dir_path.is_dir()

    def test_ensure_dir_existing(self, tmp_path):
        dir_path = tmp_path / "existing_dir"
        dir_path.mkdir()
        file_io.ensure_dir(str(dir_path)) # Should not raise error
        assert dir_path.exists()
        assert dir_path.is_dir()

    # copy_file
    def test_copy_file_success(self, tmp_path):
        src_content = "content to copy"
        src_file = tmp_path / "source.txt"
        dst_file = tmp_path / "destination.txt"
        create_dummy_file(str(src_file), src_content)

        file_io.copy_file(str(src_file), str(dst_file))
        assert dst_file.exists()
        assert dst_file.read_text(encoding='utf-8') == src_content

    def test_copy_file_src_not_found(self, tmp_path):
        with pytest.raises(FileIOException, match="Source file not found"):
            file_io.copy_file(str(tmp_path / "nonexistent_src.txt"), str(tmp_path / "dst.txt"))

    def test_copy_file_dst_exists_no_overwrite(self, tmp_path):
        src_file = tmp_path / "s.txt"
        dst_file = tmp_path / "d.txt"
        create_dummy_file(str(src_file), "src")
        create_dummy_file(str(dst_file), "dst_orig")

        with pytest.raises(FileIOException, match="Destination file already exists"):
            file_io.copy_file(str(src_file), str(dst_file), overwrite=False)

    def test_copy_file_dst_exists_overwrite(self, tmp_path):
        src_content = "new_src_content"
        src_file = tmp_path / "s_ovr.txt"
        dst_file = tmp_path / "d_ovr.txt"
        create_dummy_file(str(src_file), src_content)
        create_dummy_file(str(dst_file), "old_dst_content")

        file_io.copy_file(str(src_file), str(dst_file), overwrite=True)
        assert dst_file.read_text(encoding='utf-8') == src_content

    # read_lines and write_lines
    def test_write_read_lines_atomic(self, tmp_path):
        lines_to_write = ["line 1", " line 2 ", "", "line4"]
        file = tmp_path / "lines_atomic.txt"

        file_io.write_lines(lines_to_write, str(file), atomic=True)
        assert file.exists()

        # Test with strip=True (default)
        read_lines_stripped = file_io.read_lines(str(file))
        # SUT's write_lines adds \n, read_lines().strip() will remove leading/trailing whitespace from content of line
        expected_stripped = ["line 1", "line 2", "", "line4"]
        assert read_lines_stripped == expected_stripped

        # Test with strip=False
        # SUT's write_lines adds \n to each line. SUT's read_lines(strip=False) reads them with \n.
        expected_no_strip = [line + '\n' for line in lines_to_write]
        # We need to write the file again to ensure it has the content from lines_to_write for this specific test part
        create_dummy_file(str(file), "".join(expected_no_strip)) # Create with exact newlines
        read_lines_no_strip_actual = file_io.read_lines(str(file), strip=False)
        assert read_lines_no_strip_actual == expected_no_strip


    def test_write_read_lines_non_atomic(self, tmp_path):
        lines_to_write = ["another line", " more text"]
        file = tmp_path / "subdir_lines" / "lines_nonatomic.txt"

        file_io.write_lines(lines_to_write, str(file), atomic=False)
        assert file.exists()

        read_data = file_io.read_lines(str(file)) # Default strip=True
        expected_stripped = ["another line", "more text"]
        assert read_data == expected_stripped

    def test_read_lines_not_found(self, tmp_path):
        with pytest.raises(FileIOException, match="File not found"):
            file_io.read_lines(str(tmp_path / "no_lines_here.txt"))

    # read_struct
    def test_read_struct_success(self, tmp_path):
        # Format: int, float, 4-char string
        fmt = "=if4s"
        values = (123, 3.14, b"test")
        packed_data = struct.pack(fmt, *values)

        file = tmp_path / "struct_data.bin"
        create_dummy_file(str(file), packed_data, binary=True)

        read_values = file_io.read_struct(str(file), fmt)
        assert read_values[0] == values[0]
        assert pytest.approx(read_values[1]) == values[1]
        assert read_values[2] == values[2]

    def test_read_struct_with_offset(self, tmp_path):
        fmt = "=h" # short
        prefix = b"prefix_data"
        value = (256,)
        packed_data = prefix + struct.pack(fmt, *value)

        file = tmp_path / "struct_offset.bin"
        create_dummy_file(str(file), packed_data, binary=True)

        read_value = file_io.read_struct(str(file), fmt, offset=len(prefix))
        assert read_value == value

    def test_read_struct_not_found(self, tmp_path):
        with pytest.raises(FileIOException, match="File not found"):
            file_io.read_struct(str(tmp_path / "no_struct.bin"), "=i")

    def test_read_struct_not_enough_bytes(self, tmp_path):
        file = tmp_path / "short_struct.bin"
        create_dummy_file(str(file), b"\x01\x00", binary=True) # 2 bytes
        with pytest.raises(FileIOException, match="Could not read enough bytes"):
            file_io.read_struct(str(file), "=i") # Expects 4 bytes for int

    # safe_remove
    def test_safe_remove_existing_file(self, tmp_path):
        file = tmp_path / "to_remove.txt"
        create_dummy_file(str(file), "delete me")
        assert file.exists()
        file_io.safe_remove(str(file))
        assert not file.exists()

    def test_safe_remove_nonexistent_file(self, tmp_path):
        file = tmp_path / "already_gone.txt"
        assert not file.exists()
        file_io.safe_remove(str(file)) # Should not raise error
        assert not file.exists()

    # list_files
    def test_list_files_no_recursion_no_pattern(self, tmp_path):
        (tmp_path / "file1.txt").touch()
        (tmp_path / "file2.log").touch()
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "file3.txt").touch()

        result = sorted(file_io.list_files(str(tmp_path)))
        expected = sorted([str(tmp_path / "file1.txt"), str(tmp_path / "file2.log")])
        assert result == expected

    def test_list_files_no_recursion_with_pattern(self, tmp_path):
        (tmp_path / "file1.txt").touch()
        (tmp_path / "another.txt").touch()
        (tmp_path / "file2.log").touch()

        result = sorted(file_io.list_files(str(tmp_path), pattern=".txt")) # SUT uses 'in' for pattern
        expected = sorted([str(tmp_path / "file1.txt"), str(tmp_path / "another.txt")])
        assert result == expected

    def test_list_files_recursive_no_pattern(self, tmp_path):
        (tmp_path / "file1.txt").touch()
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "file2.txt").touch()
        deep_sub = sub / "deep"
        deep_sub.mkdir()
        (deep_sub / "file3.log").touch()

        result = sorted(file_io.list_files(str(tmp_path), recursive=True))
        expected = sorted([
            str(tmp_path / "file1.txt"),
            str(sub / "file2.txt"),
            str(deep_sub / "file3.log")
        ])
        assert result == expected

    def test_list_files_recursive_with_pattern(self, tmp_path):
        (tmp_path / "file1.txt").touch()
        (tmp_path / "data.log").touch()
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "file2.txt").touch()
        (sub / "archive.zip").touch()

        result = sorted(file_io.list_files(str(tmp_path), pattern=".txt", recursive=True))
        expected = sorted([str(tmp_path / "file1.txt"), str(sub / "file2.txt")])
        assert result == expected

    def test_list_files_dir_not_found(self, tmp_path):
        with pytest.raises(FileIOException, match="Directory not found"):
            file_io.list_files(str(tmp_path / "nonexistent_dir"))

    # file_size
    def test_file_size_success(self, tmp_path):
        content = b"12345"
        file = tmp_path / "size_test.bin"
        create_dummy_file(str(file), content, binary=True)
        assert file_io.file_size(str(file)) == len(content)

    def test_file_size_not_found(self, tmp_path):
        with pytest.raises(FileIOException, match="File not found"):
            file_io.file_size(str(tmp_path / "no_size.txt"))

    # is_json_file
    def test_is_json_file_valid(self, tmp_path):
        file = tmp_path / "valid.json"
        create_dummy_file(str(file), '{"valid": true}')
        assert file_io.is_json_file(str(file)) is True

    def test_is_json_file_invalid(self, tmp_path):
        file = tmp_path / "invalid_data.json"
        create_dummy_file(str(file), 'this is not json {')
        assert file_io.is_json_file(str(file)) is False

    def test_is_json_file_not_found(self, tmp_path):
        # SUT's is_json_file catches all exceptions and returns False
        assert file_io.is_json_file(str(tmp_path / "nonexistent.json")) is False

    # move_file
    def test_move_file_success(self, tmp_path):
        src_content = "content to move"
        src_file = tmp_path / "source_move.txt"
        dst_dir = tmp_path / "moved_to"
        # Ensure dst_dir exists for shutil.move to place file into it.
        # If dst_file is just a path and its parent dir doesn't exist, move might fail
        # depending on shutil.move's behavior (it can rename or move to dir).
        # SUT's move_file doesn't create dst parent dir.
        os.makedirs(dst_dir, exist_ok=True)
        dst_file_path = dst_dir / "destination_move.txt"
        create_dummy_file(str(src_file), src_content)

        assert src_file.exists()
        assert not dst_file_path.exists()

        file_io.move_file(str(src_file), str(dst_file_path))

        assert not src_file.exists()
        assert dst_file_path.exists()
        assert dst_file_path.read_text(encoding='utf-8') == src_content

    def test_move_file_src_not_found(self, tmp_path):
        with pytest.raises(FileIOException, match="Source file not found"):
            file_io.move_file(str(tmp_path / "no_move_src.txt"), str(tmp_path / "move_dst.txt"))

    def test_move_file_dst_exists_no_overwrite(self, tmp_path):
        src_file = tmp_path / "m_src.txt"
        dst_file = tmp_path / "m_dst.txt"
        create_dummy_file(str(src_file), "src_move_data")
        create_dummy_file(str(dst_file), "dst_move_data_orig")

        with pytest.raises(FileIOException, match="Destination file already exists"):
            file_io.move_file(str(src_file), str(dst_file), overwrite=False)

    def test_move_file_dst_exists_overwrite(self, tmp_path):
        src_content = "new_moved_content"
        src_file = tmp_path / "m_src_ovr.txt"
        dst_file = tmp_path / "m_dst_ovr.txt" # Overwriting this file
        # Ensure parent directory of dst_file exists if it's different from tmp_path
        os.makedirs(os.path.dirname(dst_file), exist_ok=True)

        create_dummy_file(str(src_file), src_content)
        create_dummy_file(str(dst_file), "old_moved_content")

        file_io.move_file(str(src_file), str(dst_file), overwrite=True)

        assert not src_file.exists() # Should be gone after move
        assert dst_file.exists()
        assert dst_file.read_text(encoding='utf-8') == src_content

    def test_write_json_creates_subdir(self, tmp_path):
        data = {"message": "test"}
        file_in_subdir = tmp_path / "new_sub" / "data.json"
        # SUT's write_json creates the directory
        # assert not os.path.exists(os.path.dirname(file_in_subdir))

        file_io.write_json(data, str(file_in_subdir))
        assert file_in_subdir.exists()
        assert json.loads(file_in_subdir.read_text()) == data

    def test_write_lines_creates_subdir(self, tmp_path):
        lines = ["hello", "world"]
        file_in_subdir = tmp_path / "another_new_sub" / "text.txt"
        # SUT's write_lines creates the directory
        # assert not os.path.exists(os.path.dirname(file_in_subdir))

        file_io.write_lines(lines, str(file_in_subdir))
        assert file_in_subdir.exists()
        assert file_in_subdir.read_text().strip().split('\n') == lines

```
I've made a slight correction to my `test_write_read_lines_atomic` logic for `strip=False` and added `from typing import Union` which I missed including in the previous block for `create_dummy_file` type hint.
Also, for `test_move_file_success` and `test_move_file_dst_exists_overwrite`, I'm ensuring the destination directory exists because `shutil.move` might behave differently if the exact destination path is a file vs. a directory (it can rename to a file in the same dir, or move into a target dir). The SUT's `move_file` doesn't create the destination parent directory.

Now, I'll try running the tests again.
