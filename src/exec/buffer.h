//
// Created by Qitong Wang on 2022/10/4.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#ifndef DSTREE_SRC_EXEC_BUFFER_H_
#define DSTREE_SRC_EXEC_BUFFER_H_

#include <unordered_map>
#include <vector>
#include <memory>
#include <fstream>

#include <spdlog/spdlog.h>

#include "global.h"
#include "config.h"
#include "eapca.h"



namespace upcite {
namespace dstree {

class Buffer {
 public:
  Buffer() = default;
  Buffer(bool is_on_disk,
         ID_TYPE capacity,
         ID_TYPE series_length,
         VALUE_TYPE *global_buffer,
         std::string dump_filepath,
         std::string load_filepath);
  ~Buffer();

  RESPONSE insert(ID_TYPE offset);
  RESPONSE flush(VALUE_TYPE *load_buffer,
                 VALUE_TYPE *flush_buffer,
                 ID_TYPE series_length);
  RESPONSE clean(bool if_remove_cache = false);

  ID_TYPE get_offset(ID_TYPE node_series_id) const { return offsets_[node_series_id]; }

  const VALUE_TYPE *get_first_series_ptr();

  const VALUE_TYPE *get_next_series_ptr();
  const VALUE_TYPE *get_series_ptr_by_id(ID_TYPE node_series_id);
  RESPONSE reset(bool reset_series_iter = true, bool free_buffer = true);

  bool is_full() const { return capacity_ > 0 && offsets_.size() == capacity_; }
  ID_TYPE size() const { return size_; }

  RESPONSE dump(std::ofstream &node_ofs) const;
  RESPONSE load(std::ifstream &node_ifs, void *ifs_buf);



// offsets_数组正是一个叶子节点内部的映射表，它有以下特点：
// 每个叶子节点有自己的offsets_数组
// 数组长度等于该节点包含的序列数量（size_）
// 数组索引是节点内的局部ID（从0开始）
// 数组值是序列在原始完整数据集中的全局位置
// 所以offsets_[node_series_id]确实就是"当前节点下的第node_series_id个序列在全局数据集中的位置"。

  // 获取当前正在处理的序列ID（在next_series_id_自增之后，需要-1）
  ID_TYPE get_current_series_id() const { 
      return next_series_id_ > 0 ? next_series_id_ - 1 : 0; 
  }
  
  // 获取当前序列的全局偏移量
  ID_TYPE get_current_offset() const {
      return get_offset(get_current_series_id());
  }

  // 重命名函数，明确返回的是全局偏移量
  ID_TYPE get_current_series_offset() const { 
      ID_TYPE local_id = next_series_id_ > 0 ? next_series_id_ - 1 : 0;
      ID_TYPE global_offset = offsets_[local_id];
      // 从dump_filepath_中提取节点ID（通常包含在文件名中）
      // printf("[DEBUG] 节点信息: offsets_大小=%zu, 当前局部ID=%d, 全局偏移量=%d\n", 
            // offsets_.size(), local_id, global_offset);
      return global_offset;
  }


 private:
  bool is_on_disk_;
  ID_TYPE capacity_;
  ID_TYPE size_;
  ID_TYPE cached_size_;
  ID_TYPE series_length_;

  std::vector<ID_TYPE> offsets_;
  VALUE_TYPE *global_buffer_, *local_buffer_;

  ID_TYPE next_series_id_;

  std::string load_filepath_; // with node.id_
  std::string dump_filepath_;
};

extern Buffer BUFFER_PLACEHOLDER;
extern std::reference_wrapper<Buffer> BUFFER_PLACEHOLDER_REF;

class BufferManager {
 public:
  BufferManager(Config &config);
  ~BufferManager();

  RESPONSE load_batch();

  VALUE_TYPE *get_series_ptr(ID_TYPE series_batch_id) const {
    return batch_load_buffer_ + config_.get().series_length_ * series_batch_id;
  }

  VALUE_TYPE *get_sketch_ptr(ID_TYPE series_batch_id) const {
    return batch_load_sketch_buffer_ + config_.get().sketch_length_ * series_batch_id;
  }

  EAPCA &get_series_eapca(ID_TYPE series_batch_id) const {
    return *batch_eapca_[series_batch_id];
  }

  RESPONSE emplace_series_eapca(std::unique_ptr<EAPCA> eapca) {
    batch_eapca_.emplace_back(std::move(eapca));
    return SUCCESS;
  }

  ID_TYPE load_buffer_size() const { return batch_nseries_; }
  bool is_fully_loaded() const { return loaded_nseries_ == config_.get().db_nseries_; }

  RESPONSE flush();
  RESPONSE clean(bool if_remove_cache = false);

  Buffer &create_node_buffer(ID_TYPE node_id);

 private:
  std::reference_wrapper<Config> config_;

  std::ifstream db_fin_;
  ID_TYPE batch_series_offset_, batch_nseries_, loaded_nseries_;
  VALUE_TYPE *batch_load_buffer_;
  std::vector<std::unique_ptr<EAPCA>> batch_eapca_;

  std::ifstream sketch_fin_;
  VALUE_TYPE *batch_load_sketch_buffer_;

  VALUE_TYPE *batch_flush_buffer_;

  std::vector<std::unique_ptr<Buffer>> node_buffers_;
  std::unordered_map<ID_TYPE, ID_TYPE> node_to_buffer_;
  std::unordered_map<ID_TYPE, ID_TYPE> buffer_to_node_;
};

}
}

#endif //DSTREE_SRC_EXEC_BUFFER_H_
