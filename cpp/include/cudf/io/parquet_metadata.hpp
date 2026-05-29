/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file parquet_metadata.hpp
 * @brief cuDF-IO freeform API
 */

#pragma once

#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/io/types.hpp>
#include <cudf/utilities/export.hpp>

#include <cstddef>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace CUDF_EXPORT cudf {
namespace io {
/**
 * @addtogroup io_types
 * @{
 * @file
 */

//! Parquet physical `Type`
using cudf::io::parquet::Type;

/**
 * @brief Schema of a parquet column, including the nested columns.
 */
struct parquet_column_schema {
 public:
  /**
   * @brief Default constructor
   *
   * This has been added since Cython requires a default constructor to create objects on stack.
   */
  explicit parquet_column_schema() : _cudf_type{data_type{type_id::EMPTY}} {}

  /**
   * @brief constructor
   *
   * @param name column name
   * @param type parquet type
   * @param children child columns (empty for non-nested types)
   * @param cudf_type cudf data type
   */
  parquet_column_schema(std::string_view name,
                        Type type,
                        std::vector<parquet_column_schema>&& children,
                        data_type cudf_type)
    : _name{name}, _type{type}, _children{std::move(children)}, _cudf_type{cudf_type}
  {
  }

  /**
   * @brief Returns parquet column name; can be empty
   *
   * @return Column name
   */
  [[nodiscard]] auto name() const { return _name; }

  /**
   * @brief Returns parquet physical type of the column.
   *
   * @return Column parquet physical type
   */
  [[nodiscard]] auto type() const { return _type; }

  /**
   * @brief Returns schemas of all child columns
   *
   * @return Children schemas
   */
  [[nodiscard]] auto const& children() const& { return _children; }

  /** @copydoc children
   * Children array is moved out of the object (rvalues only)
   *
   */
  [[nodiscard]] auto children() && { return std::move(_children); }

  /**
   * @brief Returns schema of the child with the given index
   *
   * @param idx child index
   *
   * @return Child schema
   */
  [[nodiscard]] auto const& child(int idx) const& { return children().at(idx); }

  /** @copydoc child
   * Child is moved out of the object (rvalues only)
   *
   */
  [[nodiscard]] auto child(int idx) && { return std::move(children().at(idx)); }

  /**
   * @brief Returns the number of child columns
   *
   * @return Children count
   */
  [[nodiscard]] auto num_children() const { return children().size(); }

  /**
   * @brief Returns the cudf data type for this column
   *
   * This is the resolved cudf data type mapped from the Parquet physical/logical types.
   *
   * @return cudf data type
   */
  [[nodiscard]] auto cudf_type() const { return _cudf_type; }

 private:
  std::string _name;
  // 3 types available: Physical, Converted, Logical
  Type _type;  // Physical type
  std::vector<parquet_column_schema> _children;
  data_type _cudf_type;
};

/**
 * @brief Schema of a parquet file
 */
struct parquet_schema {
 public:
  /**
   * @brief Default constructor
   *
   * This has been added since Cython requires a default constructor to create objects on stack
   */
  explicit parquet_schema() = default;

  /**
   * @brief constructor
   *
   * @param root_column_schema root column
   */
  parquet_schema(parquet_column_schema root_column_schema) : _root{std::move(root_column_schema)} {}

  /**
   * @brief Returns the schema of the struct column that contains all columns as fields
   *
   * @return Root column schema
   */
  [[nodiscard]] auto const& root() const& { return _root; }

  /** @copydoc root
   * Root column schema is moved out of the object (rvalues only)
   *
   */
  [[nodiscard]] auto root() && { return std::move(_root); }

 private:
  parquet_column_schema _root;
};

/**
 * @brief Information about content of a parquet file
 */
class parquet_metadata {
 public:
  /// Key-value metadata in the file footer
  using key_value_metadata = std::unordered_map<std::string, std::string>;
  /// Row group metadata from each RowGroup element
  using row_group_metadata = std::unordered_map<std::string, int64_t>;
  /// Column chunk metadata from each ColumnChunkMetaData element
  using column_chunk_metadata = std::unordered_map<std::string, std::vector<int64_t>>;

  /**
   * @brief Default constructor
   *
   * This has been added since Cython requires a default constructor to create objects on stack.
   */
  explicit parquet_metadata() = default;

  /**
   * @brief constructor
   *
   * @param schema parquet schema
   * @param num_rows number of rows
   * @param num_rowgroups total number of row groups
   * @param num_rowgroups_per_file number of row groups per file
   * @param file_metadata key-value metadata in the file footer
   * @param rg_metadata vector of maps containing metadata for each row group
   * @param column_chunk_metadata map of column names to vectors of `total_uncompressed_size`
   *                              metadata from all their column chunks
   */
  parquet_metadata(parquet_schema schema,
                   int64_t num_rows,
                   size_type num_rowgroups,
                   std::vector<size_type> num_rowgroups_per_file,
                   key_value_metadata file_metadata,
                   std::vector<row_group_metadata> rg_metadata,
                   column_chunk_metadata column_chunk_metadata)
    : _schema{std::move(schema)},
      _num_rows{num_rows},
      _num_rowgroups{num_rowgroups},
      _num_rowgroups_per_file{std::move(num_rowgroups_per_file)},
      _file_metadata{std::move(file_metadata)},
      _rowgroup_metadata{std::move(rg_metadata)},
      _column_chunk_metadata{std::move(column_chunk_metadata)}
  {
  }

  /**
   * @brief Returns the parquet schema
   *
   * @return parquet schema
   */
  [[nodiscard]] auto const& schema() const { return _schema; }

  /**
   * @brief Returns the number of rows of the root column
   *
   * If a file contains list columns, nested columns can have a different number of rows.
   *
   * @return Number of rows
   */
  [[nodiscard]] auto num_rows() const { return _num_rows; }

  /**
   * @brief Returns the total number of rowgroups
   *
   * @return Total number of row groups
   */
  [[nodiscard]] auto num_rowgroups() const { return _num_rowgroups; }

  /**
   * @brief Returns the number of rowgroups in each file
   *
   * @return Number of row groups per file
   */
  [[nodiscard]] auto const& num_rowgroups_per_file() const { return _num_rowgroups_per_file; }

  /**
   * @brief Returns the Key value metadata in the file footer
   *
   * @return Key value metadata as a map
   */
  [[nodiscard]] auto const& metadata() const { return _file_metadata; }

  /**
   * @brief Returns the row group metadata in the file footer
   *
   * @return Vector of row group metadata as maps
   */
  [[nodiscard]] auto const& rowgroup_metadata() const { return _rowgroup_metadata; }

  /**
   * @brief Returns a map of column names to vectors of `total_uncompressed_size` metadata from
   *        all their column chunks
   *
   * @return Map of column names to vectors of `total_uncompressed_size` metadata from all their
   *         column chunks
   */
  [[nodiscard]] auto const& columnchunk_metadata() const { return _column_chunk_metadata; }

 private:
  parquet_schema _schema;
  int64_t _num_rows;
  size_type _num_rowgroups;
  std::vector<size_type> _num_rowgroups_per_file;
  key_value_metadata _file_metadata;
  std::vector<row_group_metadata> _rowgroup_metadata;
  column_chunk_metadata _column_chunk_metadata;
};

/**
 * @brief Owning handle for parquet file footer metadata with cheap indexed views.
 */
class parquet_footer_view {
 public:
  /**
   * @brief Default constructor.
   */
  explicit parquet_footer_view() = default;

  /**
   * @brief Construct from owned parquet file metadata.
   *
   * @param file_metadatas Owned footer metadata, one element per parquet source.
   */
  explicit parquet_footer_view(std::vector<parquet::FileMetaData>&& file_metadatas)
    : _file_metadatas{std::move(file_metadatas)}
  {
  }

  /**
   * @brief Number of parquet files in this view.
   *
   * @return Number of footer metadata objects held by this view.
   */
  [[nodiscard]] std::size_t num_files() const { return _file_metadatas.size(); }

  /**
   * @brief Return file metadata for a specific source index.
   *
   * @param file_index Index of the parquet source.
   * @return File footer metadata for the selected source.
   */
  [[nodiscard]] parquet::FileMetaData const& file_metadata(std::size_t file_index) const
  {
    return _file_metadatas.at(file_index);
  }

  /**
   * @brief Return a row group view for a file and row-group index.
   *
   * @param file_index Index of the parquet source.
   * @param row_group_index Index of the row group inside the selected file.
   * @return Row group metadata for the selected file and row group.
   */
  [[nodiscard]] parquet::RowGroup const& row_group(std::size_t file_index,
                                                   std::size_t row_group_index) const
  {
    return file_metadata(file_index).row_groups.at(row_group_index);
  }

  /**
   * @brief Return a column chunk view for file, row-group, and column indexes.
   *
   * @param file_index Index of the parquet source.
   * @param row_group_index Index of the row group inside the selected file.
   * @param column_index Index of the column chunk inside the selected row group.
   * @return Column chunk metadata for the selected file, row group, and column.
   */
  [[nodiscard]] parquet::ColumnChunk const& column_chunk(std::size_t file_index,
                                                         std::size_t row_group_index,
                                                         std::size_t column_index) const
  {
    return row_group(file_index, row_group_index).columns.at(column_index);
  }

  /**
   * @brief Return a sorting column view for file, row-group, and sorting-column indexes.
   *
   * @param file_index Index of the parquet source.
   * @param row_group_index Index of the row group inside the selected file.
   * @param sorting_column_index Index of the sorting-column metadata entry.
   * @return Sorting-column metadata for the selected row group.
   */
  [[nodiscard]] parquet::SortingColumn const& sorting_column(std::size_t file_index,
                                                             std::size_t row_group_index,
                                                             std::size_t sorting_column_index) const
  {
    auto const& sorting_columns = row_group(file_index, row_group_index).sorting_columns;
    if (not sorting_columns.has_value()) {
      throw std::out_of_range("Row group has no sorting columns");
    }
    return sorting_columns.value().at(sorting_column_index);
  }

  /**
   * @brief Release ownership of all file metadata.
   *
   * @return Moved-out vector of owned file footer metadata.
   */
  [[nodiscard]] std::vector<parquet::FileMetaData> release() &&
  {
    return std::move(_file_metadatas);
  }

 private:
  std::vector<parquet::FileMetaData> _file_metadatas;
};

/**
 * @brief Reads metadata of parquet dataset
 *
 * @ingroup io_readers
 *
 * @param src_info Dataset source information
 *
 * @return parquet_metadata with parquet schema, number of rows, number of row groups and key-value
 * metadata
 */
parquet_metadata read_parquet_metadata(source_info const& src_info);

/**
 * @brief Constructs FileMetaData objects from parquet dataset
 *
 * @ingroup io_readers
 *
 * @param sources Input `datasource` objects to read the dataset from
 *
 * @return List of FileMetaData objects, one per parquet source
 */
std::vector<parquet::FileMetaData> read_parquet_footers(
  cudf::host_span<std::unique_ptr<cudf::io::datasource> const> sources);

/**
 * @brief Constructs a view handle over parquet file footers.
 *
 * @ingroup io_readers
 *
 * @param sources Input `datasource` objects to read the dataset from
 *
 * @return Handle owning one FileMetaData object per parquet source
 */
parquet_footer_view read_parquet_footers_view(
  cudf::host_span<std::unique_ptr<cudf::io::datasource> const> sources);

/** @} */  // end of group
}  // namespace io
}  // namespace CUDF_EXPORT cudf
