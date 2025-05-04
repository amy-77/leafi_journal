//
// Created by Qitong Wang on 2022/10/4.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#include "buffer.h"

#include <immintrin.h>
#include <iostream>
#include <utility>

//#include <boost/format.hpp>
#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

namespace dstree = upcite::dstree;

dstree::Buffer BUFFER_PLACEHOLDER = dstree::Buffer();
std::reference_wrapper<dstree::Buffer> BUFFER_PLACEHOLDER_REF = std::ref(BUFFER_PLACEHOLDER);

dstree::Buffer::Buffer(bool is_on_disk,
                       ID_TYPE capacity,
                       ID_TYPE series_length,
                       VALUE_TYPE *global_buffer,
                       std::string dump_filepath,
                       std::string load_filepath) :
    is_on_disk_(is_on_disk),
    capacity_(capacity),
    series_length_(series_length),
    global_buffer_(global_buffer),
    local_buffer_(nullptr),
    dump_filepath_(std::move(dump_filepath)),
    load_filepath_(std::move(load_filepath)),
    size_(0),
    cached_size_(0),
    next_series_id_(0) {
  offsets_.reserve(16);
}

dstree::Buffer::~Buffer() {
  if (local_buffer_ != nullptr) {
    std::free(local_buffer_);
    local_buffer_ = nullptr;
  }
}


const VALUE_TYPE *dstree::Buffer::get_first_series_ptr() {
  next_series_id_ = 0;

  if (is_on_disk_) {
    if (cached_size_ == 0) {
      assert(local_buffer_ == nullptr);

      ID_TYPE local_buffer_nbytes = static_cast<ID_TYPE>(sizeof(VALUE_TYPE)) * series_length_ * size_;
      local_buffer_ = static_cast<VALUE_TYPE *>(aligned_alloc(sizeof(__m256), local_buffer_nbytes));

      if (!fs::exists(load_filepath_)) {
        spdlog::error("node data file {:s} does not exist", load_filepath_);

        return nullptr;
      }

      std::ifstream fin(load_filepath_, std::ios::in | std::ios::binary);
      if (!fin.good()) {
        spdlog::error("node data file {:s} cannot open", load_filepath_);

        return nullptr;
      }

      fin.read(reinterpret_cast<char *>(local_buffer_), local_buffer_nbytes);

      if (fin.fail()) {
        spdlog::error("node buffer cannot read {:d} bytes from {:s}", local_buffer_nbytes, load_filepath_);

        return nullptr;
      }

      fin.close();

      cached_size_ = size_;
    }

    return local_buffer_ + series_length_ * next_series_id_++;
  } else {
    return global_buffer_ + series_length_ * offsets_[next_series_id_++];
  }

  return nullptr;
}


const VALUE_TYPE *dstree::Buffer::get_next_series_ptr() {
#ifdef DEBUG
#ifndef DEBUGGED
  spdlog::debug("buffer is_disk {:b} next_i {:d} size_ {:d}/{:d}",
                is_on_disk_, next_series_id_, size_, capacity_);
#endif
#endif

  // printf("[DEBUG] buffer is_disk %d next_i %d size_ %d/%d\n", 
  //       (int)is_on_disk_,   // {:b} → %d (布尔转整型)
  //       next_series_id_,    // {:d} → %d (整型)
  //       size_,              // {:d} → %d
  //       capacity_);         // {:d} → %d
  // capacity: 每个叶子节点的最大序列数
  // size_: 当前叶子节点中已经存储的序列数
  // next_series_id_: 当前正在处理的序列ID

  if (is_on_disk_) {
    // 磁盘模式下进行安全检查
    if (local_buffer_ == nullptr) {
      printf("错误: get_next_series_ptr中local_buffer_为空\n");
      return nullptr;
    }
    
    if (cached_size_ != size_) {
      printf("警告: get_next_series_ptr中cached_size_(%d)不等于size_(%d)\n", cached_size_, size_);
    }

    // 注意：next_series_id_ == size_是正常的退出条件，不应报错
    if (next_series_id_ < 0) {
      printf("错误: get_next_series_ptr中next_series_id_(%d)为负数\n", next_series_id_);
      return nullptr;
    }

    // 这是正常的循环终止检查
    if (next_series_id_ >= size_) {
      // 已到达序列末尾，返回nullptr是正常的
      return nullptr;
    }

    VALUE_TYPE* ptr = local_buffer_ + series_length_ * next_series_id_++;
    
    // 检查指针的有效性
    if (ptr == nullptr || ptr < local_buffer_ || ptr >= local_buffer_ + series_length_ * size_) {
      printf("警告: get_next_series_ptr返回无效指针: %p\n", (void*)ptr);
      return nullptr;
    }
    
    return ptr;
  } else {
    // 内存模式下进行安全检查
    if (global_buffer_ == nullptr) {
      printf("错误: get_next_series_ptr中global_buffer_为空\n");
      return nullptr;
    }

    if (offsets_.empty()) {
      printf("错误: get_next_series_ptr中offsets_数组为空\n");
      return nullptr;
    }

    // 注意：next_series_id_ == size_是正常的退出条件，不应报错
    if (next_series_id_ < 0) {
      printf("错误: get_next_series_ptr中next_series_id_(%d)为负数\n", next_series_id_);
      return nullptr;
    }
    
    // 这是正常的循环终止检查
    if (next_series_id_ >= size_ || next_series_id_ >= offsets_.size()) {
      // 已到达序列末尾，返回nullptr是正常的
      return nullptr;
    }

    // 获取偏移量并检查
    ID_TYPE offset = offsets_[next_series_id_];
    // printf("\n[DEBUG] 内存地址计算详情:\n");
    // printf("  global_buffer_地址: %p\n", (void*)global_buffer_);
    // printf("  序列长度(series_length_): %d\n", series_length_);
    // printf("  当前序列偏移量(offset): %d\n", offset);
    // printf("  next_series_id_: %d\n", next_series_id_-1); // 已经自增过
    // printf("  计算的指针地址: %p\n", (void*)(global_buffer_ + series_length_ * offset));
    // printf("  从计算地址获取的第一个值: %f\n\n", *(global_buffer_ + series_length_ * offset));

    if (offset < 0) {
      printf("错误: get_next_series_ptr中offset值为负数: %d\n", offset);
      next_series_id_++; // 仍然递增以避免无限循环
      return nullptr;
    }
    
    VALUE_TYPE* ptr = global_buffer_ + series_length_ * offset;
    next_series_id_++;
    
    // 检查指针的有效性
    if (ptr == nullptr || ptr < global_buffer_) {
      printf("警告: get_next_series_ptr返回无效指针: %p\n", (void*)ptr);
      return nullptr;
    }
    return ptr;
  }
}

const VALUE_TYPE *dstree::Buffer::get_series_ptr_by_id(ID_TYPE node_series_id) {
  // 首先检查node_series_id是否有效
  if (node_series_id < 0) {
    printf("错误: get_series_ptr_by_id中node_series_id为负数: %d\n", node_series_id);
    return nullptr;
  }

  if (is_on_disk_) {
    if (cached_size_ == 0) {
      if (local_buffer_ != nullptr) {
        printf("警告: cached_size_为0但local_buffer_不为空\n");
        std::free(local_buffer_);
        local_buffer_ = nullptr;
      }

      // 添加大小检查，防止申请过大内存
      if (size_ <= 0) {
        printf("错误: get_series_ptr_by_id中size_值为0或负数：%d，无法分配内存\n", size_);
        return nullptr;
      }

      ID_TYPE local_buffer_nbytes = static_cast<ID_TYPE>(sizeof(VALUE_TYPE)) * series_length_ * size_;
      if (local_buffer_nbytes <= 0) {
        printf("错误: get_series_ptr_by_id中本地缓冲区大小计算异常：%d字节\n", local_buffer_nbytes);
        return nullptr;
      }

      try {
        local_buffer_ = static_cast<VALUE_TYPE *>(aligned_alloc(sizeof(__m256), local_buffer_nbytes));
        if (local_buffer_ == nullptr) {
          printf("错误: get_series_ptr_by_id中内存分配失败，无法分配%d字节\n", local_buffer_nbytes);
          return nullptr;
        }
      } catch (const std::exception& e) {
        printf("异常: get_series_ptr_by_id中内存分配失败：%s\n", e.what());
        return nullptr;
      } catch (...) {
        printf("未知异常: get_series_ptr_by_id中内存分配失败\n");
        return nullptr;
      }

      if (!fs::exists(load_filepath_)) {
        spdlog::error("node file {:s} does not exist", load_filepath_);
        printf("错误：get_series_ptr_by_id中节点文件 %s 不存在\n", load_filepath_.c_str());
        std::free(local_buffer_);
        local_buffer_ = nullptr;
        return nullptr;
      }

      std::ifstream fin(load_filepath_, std::ios::in | std::ios::binary);
      if (!fin.good()) {
        spdlog::error("node file {:s} cannot open", load_filepath_);
        printf("错误：get_series_ptr_by_id中无法打开节点文件 %s\n", load_filepath_.c_str());
        std::free(local_buffer_);
        local_buffer_ = nullptr;
        return nullptr;
      }

      fin.read(reinterpret_cast<char *>(local_buffer_), local_buffer_nbytes);

      if (fin.fail()) {
        spdlog::error("node buffer cannot read {:d} bytes from {:s}", local_buffer_nbytes, load_filepath_);
        printf("错误：get_series_ptr_by_id中无法从 %s 读取 %d 字节\n", load_filepath_.c_str(), local_buffer_nbytes);
        std::free(local_buffer_);
        local_buffer_ = nullptr;
        return nullptr;
      }
      fin.close();
      cached_size_ = size_;
    }

    if (local_buffer_ == nullptr) {
      printf("错误：get_series_ptr_by_id中local_buffer_为空\n");
      return nullptr;
    }

    if (node_series_id >= size_) {
      printf("错误：get_series_ptr_by_id中node_series_id(%d)超出范围[0,%d)\n", node_series_id, size_);
      return nullptr;
    }

    VALUE_TYPE* ptr = local_buffer_ + series_length_ * node_series_id;
    
    // 检查指针的有效性
    if (ptr == nullptr || ptr < local_buffer_ || ptr >= local_buffer_ + series_length_ * size_) {
      printf("警告：get_series_ptr_by_id返回无效磁盘指针: %p\n", (void*)ptr);
      return nullptr;
    }
    return ptr;

  } else {
    // 内存模式下的安全检查
    if (global_buffer_ == nullptr) {
      printf("错误：get_series_ptr_by_id中global_buffer_为空\n");
      return nullptr;
    }

    if (offsets_.empty()) {
      printf("错误：get_series_ptr_by_id中offsets_数组为空\n");
      return nullptr;
    }

    if (node_series_id >= size_ || node_series_id >= offsets_.size()) {
      printf("错误：get_series_ptr_by_id中node_series_id(%d)超出范围[0,%d)或[0,%zu)\n", 
             node_series_id, size_, offsets_.size());
      return nullptr;
    }

    // 获取偏移量并检查
    ID_TYPE offset = offsets_[node_series_id];
    if (offset < 0) {
      printf("错误：get_series_ptr_by_id中offset值为负数: %d\n", offset);
      return nullptr;
    }
    
    VALUE_TYPE* ptr = global_buffer_ + series_length_ * offset;
    
    // 检查指针的有效性
    if (ptr == nullptr || ptr < global_buffer_) {
      printf("警告：get_series_ptr_by_id返回无效内存指针: %p\n", (void*)ptr);
      return nullptr;
    }
    
    return ptr;
  }

  return nullptr;
}




RESPONSE dstree::Buffer::reset(bool reset_series_iter, bool free_buffer) {
  if (local_buffer_ != nullptr && free_buffer) {
    std::free(local_buffer_);
    local_buffer_ = nullptr;
    cached_size_ = 0;
  }

  if (reset_series_iter) {
    next_series_id_ = 0;
  }

  return SUCCESS;
}

RESPONSE dstree::Buffer::insert(ID_TYPE offset) {
  if (size_ >= capacity_) {
    spdlog::error("node {:s} buffer nseries > capacity", fs::path(dump_filepath_).filename().string());
    return FAILURE;
  }

  // TODO explicitly managing ID_TYPE * resulted in unexpected change of values in the middle
  offsets_.push_back(offset);

  size_ += 1;
  cached_size_ += 1;

#ifdef DEBUG
#ifndef DEBUGGED
  if (logger != nullptr) {
    MALAT_LOG(logger->logger, trivial::debug)
      << boost::format("%d / %d = %d == %d")
          % size()
          % capacity_
          % offset
          % offsets_[size() - 1];
  }
#endif
#endif

  return SUCCESS;
}

RESPONSE dstree::Buffer::flush(VALUE_TYPE *load_buffer, VALUE_TYPE *flush_buffer, ID_TYPE series_length) {
  if (cached_size_ > 0) {
    auto series_nbytes = static_cast<ID_TYPE>(sizeof(VALUE_TYPE)) * series_length;

    for (ID_TYPE i = 0; i < cached_size_; ++i) {
      std::memcpy(flush_buffer + series_length * i, load_buffer + series_length * offsets_[i], series_nbytes);
    }

    std::ofstream fout(dump_filepath_, std::ios::binary | std::ios_base::app);
    fout.write(reinterpret_cast<char *>(flush_buffer), series_nbytes * cached_size_);
    fout.close();

    cached_size_ = 0;
  }

  return SUCCESS;
}

RESPONSE dstree::Buffer::clean(bool if_remove_cache) {
  offsets_.clear();

  size_ = 0;
  cached_size_ = 0;

  if (if_remove_cache) {
//    offsets_.shrink_to_fit();
    std::vector<ID_TYPE>().swap(offsets_);
    capacity_ = 0;

    if (local_buffer_ != nullptr) {
      free(local_buffer_);
      local_buffer_ = nullptr;
    }
  }

  return SUCCESS;
}

RESPONSE dstree::Buffer::dump(std::ofstream &node_ofs) const {
  node_ofs.write(reinterpret_cast<const char *>(&size_), sizeof(ID_TYPE));
  assert(offsets_.size() == size_);

//  spdlog::debug("buffer dump size {:d} offsets.size {:d}", size_, offsets_.size());

  if (size_ > 0) {
    node_ofs.write(reinterpret_cast<const char *>(offsets_.data()), sizeof(ID_TYPE) * offsets_.size());
  }

  return SUCCESS;
}

RESPONSE dstree::Buffer::load(std::ifstream &node_ifs, void *ifs_buf) {
  auto ifs_id_buf = reinterpret_cast<ID_TYPE *>(ifs_buf);

  ID_TYPE read_nbytes = sizeof(ID_TYPE);
  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  size_ = ifs_id_buf[0];

//  spdlog::debug("buffer load size {:d} offsets.size {:d}", size_, offsets_.size());

  if (size_ > 0) {
    read_nbytes = sizeof(ID_TYPE) * size_;
    node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
    offsets_.insert(offsets_.begin(), ifs_id_buf, ifs_id_buf + size_);
  } else {
    // TODO fix the size_ = -4294967296 problem when dump size_ = 0
    size_ = 0;
  }

  return SUCCESS;
}

dstree::BufferManager::BufferManager(dstree::Config &config) :
    config_(config),
    batch_series_offset_(0),
    loaded_nseries_(0),
    batch_flush_buffer_(nullptr) {
  batch_nseries_ = config.batch_load_nseries_;
  auto batch_nbytes = static_cast<ID_TYPE>(sizeof(VALUE_TYPE)) * config.series_length_ * batch_nseries_;
  batch_load_buffer_ = static_cast<VALUE_TYPE *>(aligned_alloc(sizeof(__m256), batch_nbytes));

  node_buffers_.reserve(config.default_nbuffer_);

  node_to_buffer_.reserve(config.default_nbuffer_);
  buffer_to_node_.reserve(config.default_nbuffer_);

  if (config_.get().is_sketch_provided_) {
    auto batch_sketch_nbytes = static_cast<ID_TYPE>(sizeof(VALUE_TYPE)) * config.sketch_length_ * batch_nseries_;
    batch_load_sketch_buffer_ = static_cast<VALUE_TYPE *>(malloc(batch_sketch_nbytes));
  }
}

dstree::BufferManager::~BufferManager() {
  if (batch_load_buffer_) {
    std::free(batch_load_buffer_);
    batch_load_buffer_ = nullptr;
  }

  if (batch_flush_buffer_) {
    std::free(batch_flush_buffer_);
    batch_flush_buffer_ = nullptr;
  }

  if (db_fin_.is_open()) {
    db_fin_.close();
  }

  if (config_.get().is_sketch_provided_) {
    if (batch_load_sketch_buffer_) {
      std::free(batch_load_sketch_buffer_);
      batch_load_sketch_buffer_ = nullptr;
    }

    if (sketch_fin_.is_open()) {
      sketch_fin_.close();
    }
  }
}

dstree::Buffer &dstree::BufferManager::create_node_buffer(ID_TYPE node_id) {
  auto buffer_id = static_cast<ID_TYPE>(node_buffers_.size());
  std::string buffer_filepath =
      config_.get().dump_data_folderpath_ + std::to_string(node_id) + config_.get().index_dump_file_postfix_;

  std::string load_filepath = buffer_filepath;
  if (config_.get().to_load_index_) {
    load_filepath =
        config_.get().load_data_folderpath_ + std::to_string(node_id) + config_.get().index_dump_file_postfix_;
  }

  node_buffers_.emplace_back(std::make_unique<dstree::Buffer>(
      config_.get().on_disk_,
      config_.get().leaf_max_nseries_,
      config_.get().series_length_,
      batch_load_buffer_,
      buffer_filepath,
      load_filepath));

  node_to_buffer_[node_id] = buffer_id;
  buffer_to_node_[buffer_id] = node_id;

  return *node_buffers_[buffer_id];
}

RESPONSE dstree::BufferManager::load_batch() {
  printf("[DEBUG] Loading a batch of data...\n");
  // 加载数据的逻辑
  if (loaded_nseries_ == 0) {
    if (!fs::exists(config_.get().db_filepath_)) {
      spdlog::error("database filepath does not exist = {:s}", config_.get().db_filepath_);

      return FAILURE;
    }

    db_fin_.open(config_.get().db_filepath_, std::ios::in | std::ios::binary);
    if (!db_fin_.good()) {
      spdlog::error("database filepath cannot open = {:s}", config_.get().db_filepath_);

      return FAILURE;
    }

    if (config_.get().is_sketch_provided_) {
      if (!fs::exists(config_.get().train_sketch_filepath_)) {
        spdlog::error("database sketch filepath does not exist = {:s}", config_.get().train_sketch_filepath_);

        return FAILURE;
      }

      sketch_fin_.open(config_.get().train_sketch_filepath_, std::ios::in | std::ios::binary);
      if (!sketch_fin_.good()) {
        spdlog::error("database sketch filepath cannot open = {:s}", config_.get().train_sketch_filepath_);

        return FAILURE;
      }
    }
  } else if (loaded_nseries_ >= config_.get().db_nseries_) {
    return FAILURE;
  }

  if (loaded_nseries_ + config_.get().batch_load_nseries_ > config_.get().db_nseries_) {
    batch_nseries_ = config_.get().db_nseries_ - loaded_nseries_;
  }
  auto batch_nbytes = static_cast<ID_TYPE>(sizeof(VALUE_TYPE)) * config_.get().series_length_ * batch_nseries_;

  batch_series_offset_ = loaded_nseries_;
  auto batch_bytes_offset =
      static_cast<ID_TYPE>(sizeof(VALUE_TYPE)) * config_.get().series_length_ * batch_series_offset_;

  db_fin_.seekg(batch_bytes_offset);
  db_fin_.read(reinterpret_cast<char *>(batch_load_buffer_), batch_nbytes);

  if (db_fin_.fail()) {
    spdlog::error("cannot read {:d} bytes from {:s} at {:d}",
                  batch_nbytes, config_.get().db_filepath_, batch_bytes_offset);

    return FAILURE;
  }

  if (config_.get().is_sketch_provided_) {
    auto batch_sketch_nbytes = static_cast<ID_TYPE>(sizeof(VALUE_TYPE)) * config_.get().sketch_length_ * batch_nseries_;
    auto batch_sketch_bytes_offset =
        static_cast<ID_TYPE>(sizeof(VALUE_TYPE)) * config_.get().sketch_length_ * batch_series_offset_;

    sketch_fin_.seekg(batch_sketch_bytes_offset);
    sketch_fin_.read(reinterpret_cast<char *>(batch_load_sketch_buffer_), batch_sketch_nbytes);

    if (sketch_fin_.fail()) {
      spdlog::error("cannot read {:d} bytes from {:s} at {:d}",
                    batch_sketch_nbytes, config_.get().train_sketch_filepath_, batch_sketch_bytes_offset);

      return FAILURE;
    }
  }

  loaded_nseries_ += batch_nseries_;

  return SUCCESS;
}

RESPONSE dstree::BufferManager::flush() {
  if (batch_flush_buffer_ == nullptr) {
    auto batch_flush_nbytes = static_cast<ID_TYPE>(sizeof(VALUE_TYPE)) *
        config_.get().series_length_ * config_.get().leaf_max_nseries_;
    batch_flush_buffer_ = static_cast<VALUE_TYPE *>(std::malloc(batch_flush_nbytes));
  }

  for (const auto &buffer : node_buffers_) {
    buffer->flush(batch_load_buffer_, batch_flush_buffer_, config_.get().series_length_);
  }

  // TODO flush sketches

  return SUCCESS;
}

RESPONSE dstree::BufferManager::clean(bool if_remove_cache) {
  for (const auto &buffer : node_buffers_) {
    buffer->clean(if_remove_cache);
  }

  return SUCCESS;
}
