

// 执行暴力搜索
RESPONSE dstree::Index::profile(ID_TYPE query_id, VALUE_TYPE *query_ptr, VALUE_TYPE *sketch_ptr,
                                dstree::Answers *results){ // 添加参数
  VALUE_TYPE *route_ptr = query_ptr;
  if (config_.get().is_sketch_provided_){
    route_ptr = sketch_ptr;
  }

  // 修改：使用传入的results对象或创建新的
  std::shared_ptr<dstree::Answers> answers;
  if (results != nullptr){
    // printf("results != nullptr, 使用传入的results对象\n");
    //实际走的是这个分支
    answers = std::shared_ptr<dstree::Answers>(results, [](dstree::Answers *) {}); // 非拥有指针
  } else {
    // printf("results == nullptr, 创建新的answers对象\n");
    answers = std::make_shared<dstree::Answers>(config_.get().n_nearest_neighbor_, query_id);
  }

  if (config_.get().require_neurofilter_){
    filter_query_tsr_ = torch::from_blob(query_ptr,
                                         {1, config_.get().series_length_},
                                         torch::TensorOptions().dtype(TORCH_VALUE_TYPE));
    filter_query_tsr_ = filter_query_tsr_.to(*device_);
  }

  ID_TYPE visited_node_counter = 0, visited_series_counter = 0;
  leaf_min_heap_.push(std::make_tuple(std::ref(*root_), root_->cal_lower_bound_EDsquare(route_ptr)));

  std::reference_wrapper<dstree::Node> node_to_visit = std::ref(*(dstree::Node *)nullptr);
  VALUE_TYPE node2visit_lbdistance;

  while (!leaf_min_heap_.empty()){
    std::tie(node_to_visit, node2visit_lbdistance) = leaf_min_heap_.top();
    leaf_min_heap_.pop();

    if (node_to_visit.get().is_leaf()){
      if (visited_node_counter < config_.get().search_max_nnode_ && visited_series_counter < config_.get().search_max_nseries_){
        //is_bsf 用于判断lb_distance是否小于minbsf
        if (answers->is_bsf(node2visit_lbdistance)) {
          node_to_visit.get().search(query_ptr, query_id, *answers, visited_node_counter, visited_series_counter);
        }
        visited_node_counter += 1;
        visited_series_counter += node_to_visit.get().get_size();
      }

    } else {
      for (auto child_node : node_to_visit.get()){
        VALUE_TYPE child_lower_bound_EDsquare = child_node.get().cal_lower_bound_EDsquare(route_ptr);
        leaf_min_heap_.push(std::make_tuple(child_node, child_lower_bound_EDsquare));
      }
    }
  }

  // spdlog::info("Ground Truth: query {:d} visited {:d} nodes {:d} series", query_id, visited_node_counter, visited_series_counter);

  // 修改：只在未传入results时打印结果
  if (results == nullptr){
    ID_TYPE nnn_to_return = config_.get().n_nearest_neighbor_;
    while (!answers->empty()){
      auto answer = answers->pop_answer();
      if (answer.node_id_ > 0){
        printf("query %d nn %d = %.3f, node %d, db_global_offset_ %d\n",
               query_id, nnn_to_return, answer.nn_dist_, answer.node_id_, answer.global_offset_);
        spdlog::info("query {} nn {} = {:.3f}, node {}, db_global_offset_ {}",
               query_id, nnn_to_return, answer.nn_dist_, answer.node_id_, answer.global_offset_);
      } else {
        printf("query %d nn %d = %.3f, db_global_offset_ %d\n",
               query_id, nnn_to_return, answer.nn_dist_, answer.global_offset_);
        spdlog::info("query {} nn {} = {:.3f}, node {}, db_global_offset_ {}",
               query_id, nnn_to_return, answer.nn_dist_, answer.node_id_, answer.global_offset_);
      }
      nnn_to_return -= 1;
    }
    if (nnn_to_return > 0){
      return FAILURE;
    }
  }

  return SUCCESS;
}



// 执行带过滤器的实际搜： 查找给定query id后的全局最近邻
RESPONSE dstree::Index::search(ID_TYPE query_id, VALUE_TYPE *query_ptr, VALUE_TYPE *sketch_ptr, dstree::Answers *results){ // 添加参数
  VALUE_TYPE *route_ptr = query_ptr;
  if (config_.get().is_sketch_provided_){
    route_ptr = sketch_ptr;
  }

  // 修改：使用传入的results对象或创建新的
  std::shared_ptr<dstree::Answers> answers;
  if (results != nullptr){
    answers = std::shared_ptr<dstree::Answers>(results, [](dstree::Answers *) {}); // 非拥有指针
  } else {
    answers = std::make_shared<dstree::Answers>(config_.get().n_nearest_neighbor_, query_id);
  }

  if (config_.get().require_neurofilter_){
    filter_query_tsr_ = torch::from_blob(query_ptr,
                                         {1, config_.get().series_length_},
                                         torch::TensorOptions().dtype(TORCH_VALUE_TYPE));
    filter_query_tsr_ = filter_query_tsr_.to(*device_);
  }

  //从根节点开始，逐步向下遍历树结构，直到找到包含目标查询序列的叶子节点。
  std::reference_wrapper<dstree::Node> resident_node = std::ref(*root_);
  while (!resident_node.get().is_leaf()){
    //这步快速找到最可能包含相似序列的初始叶子节点
    resident_node = resident_node.get().route(route_ptr);
  }
  
  // 阶段一： 近似搜索，查找当前节点resident_node下距离query_ptr最近距离，更新bsf
  ID_TYPE visited_node_counter = 0, visited_series_counter = 0;
  ID_TYPE nfpruned_node_counter = 0, nfpruned_series_counter = 0;
  resident_node.get().search(query_ptr, query_id, *answers, visited_node_counter, visited_series_counter);
  //  filter_collect: VALUE_TYPE local_nn_distance = resident_node.get().search(series_ptr, query_id, m256_fetch_cache, answer.get());

  //阶段二：精确搜索：对于同一个query，使用优先队列(leaf_min_heap_)按下界距离从小到大检查其他节点
  if (config_.get().is_exact_search_){

    leaf_min_heap_.push(std::make_tuple(std::ref(*root_), root_->cal_lower_bound_EDsquare(route_ptr)));
    std::reference_wrapper<dstree::Node> node_to_visit = std::ref(*(dstree::Node *)nullptr);
    VALUE_TYPE node2visit_lbdistance;

    while (!leaf_min_heap_.empty()){
      std::tie(node_to_visit, node2visit_lbdistance) = leaf_min_heap_.top();
      leaf_min_heap_.pop();

      if (node_to_visit.get().is_leaf()){
        if (visited_node_counter < config_.get().search_max_nnode_ && visited_series_counter < config_.get().search_max_nseries_){
          
          if (node_to_visit.get().get_id() != resident_node.get().get_id()){
            //  利用lbdistance判断：lb<minbsf 继续搜索？ 问：lb>minbsf, 此时true>lb>minbsf, 不需要继续搜
            if (answers->is_bsf(node2visit_lbdistance)) {
              // if (node_to_visit.get().has_active_filter()){
                // VALUE_TYPE predicted_nn_distance = node_to_visit.get().filter_infer_calibrated(filter_query_tsr_);
                // if (predicted_nn_distance > answers->get_bsf()){
                //   // 如果预测的nn距离大于bsf距离，则进行过滤
                //   printf("剪枝: 节点%d: 预测距离=%.3f, BSF=%.3f\n", 
                //          node_to_visit.get().get_id(), predicted_nn_distance, answers->get_bsf());
                //   spdlog::info("剪枝: 节点{}: 预测距离={:.3f}, BSF={:.3f}", 
                //               node_to_visit.get().get_id(), predicted_nn_distance, answers->get_bsf());
                //   nfpruned_node_counter += 1;
                //   nfpruned_series_counter += node_to_visit.get().get_size();
                // } else {
                //   // 如果pred-error < bsf，则不进行剪枝
                //   node_to_visit.get().search(query_ptr, query_id, *answers, visited_node_counter,
                //                              visited_series_counter);
                // }

              // } else {
              //   //对于没有激活过滤器的节点，直接进行搜索
              //   node_to_visit.get().search(query_ptr, query_id, *answers, visited_node_counter,
              //                              visited_series_counter);
              // }
              //暂时修改用于检查
              node_to_visit.get().search(query_ptr, query_id, *answers, visited_node_counter, visited_series_counter);
            }
          }
        }

      } else {
        // 对于非叶子节点，将子节点加入优先队列
        for (auto child_node : node_to_visit.get()){
          VALUE_TYPE child_lower_bound_EDsquare = child_node.get().cal_lower_bound_EDsquare(route_ptr);
          leaf_min_heap_.push(std::make_tuple(child_node, child_lower_bound_EDsquare));
        }
      }

    }
  }

  // spdlog::info("Actual: query {:d} visited {:d} nodes {:d} series, filtered {:d} nodes {:d} series", query_id, visited_node_counter, visited_series_counter, nfpruned_node_counter, nfpruned_series_counter);

  // 修改：只在未传入results时打印结果
  if (results == nullptr){
    printf("----------search:results == nullptr, 打印结果----------\n");
    ID_TYPE nnn_to_return = config_.get().n_nearest_neighbor_;
    while (!answers->empty()) {
      auto answer = answers->pop_answer();
      if (answer.node_id_ > 0){
        printf("query %d nn %d = %.3f, node %d, db_global_offset_ %d\n",
               query_id, nnn_to_return, answer.nn_dist_, answer.node_id_, answer.global_offset_);
        spdlog::info("query {} nn {} = {:.3f}, node {}, db_global_offset_ {}",
               query_id, nnn_to_return, answer.nn_dist_, answer.node_id_, answer.global_offset_);
      } else {
        printf("query %d nn %d = %.3f, db_global_offset_ %d\n", query_id, nnn_to_return, answer.nn_dist_, answer.global_offset_);
        spdlog::info("query {} nn {} = {:.3f}, node {}, db_global_offset_ {}",
               query_id, nnn_to_return, answer.nn_dist_, answer.node_id_, answer.global_offset_);
      }
      nnn_to_return -= 1;
    }
    if (nnn_to_return > 0){
      return FAILURE;
    }
  }

  return SUCCESS;
}


