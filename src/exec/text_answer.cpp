// test_answers.cpp
#include <iostream>
#include "answer.h"  // 确保包含你的 Answers 类定义

namespace dstree = upcite::dstree;
namespace constant = upcite::constant;

// 辅助函数：打印节点的 KNN 分布
void print_knn_distribution(const std::unordered_map<ID_TYPE, std::pair<ID_TYPE, VALUE_TYPE>>& distribution) {
    for (const auto& entry : distribution) {
        ID_TYPE node_id = entry.first;
        ID_TYPE count = entry.second.first;
        VALUE_TYPE min_dist = entry.second.second;
        std::cout << "节点 " << node_id << ": " << count << " 个结果，最小距离 = " << min_dist << std::endl;
    }
}

int main() {
    // ====================== 测试 1：基本功能 ======================
    std::cout << "===== 测试 1：基本功能 =====" << std::endl;
    dstree::Answers answers(3, 0);  // 容量为3，查询ID为0

    // 添加结果
    answers.push_bsf(1.5, 100, 0);  // 距离1.5，节点100，查询0
    answers.push_bsf(2.0, 101, 0);  // 距离2.0，节点101，查询0
    answers.push_bsf(0.8, 102, 0);  // 距离0.8，节点102，查询0

    // 获取当前最佳距离（应为最大的最小距离，即1.5）
    std::cout << "当前最佳距离（BSF）: " << answers.get_bsf() << std::endl;

    // 打印所有结果
    std::cout << "当前Top K结果：" << std::endl;
    auto topk = answers.get_current_topk();
    for (const auto& ans : topk) {
        std::cout << "距离: " << ans.nn_dist_ << ", 节点: " << ans.node_id_ << std::endl;
    }

    // ====================== 测试 2：深拷贝功能 ======================
    std::cout << "\n===== 测试 2：深拷贝功能 =====" << std::endl;
    dstree::Answers copy(answers);  // 调用拷贝构造函数

    // 修改原对象
    answers.push_bsf(3.0, 103, 0);  // 超出容量3，会被忽略

    // 检查原对象和副本的结果数量
    std::cout << "原对象结果数: " << answers.get_current_topk().size() << std::endl;
    std::cout << "副本结果数: " << copy.get_current_topk().size() << std::endl;

    // ====================== 测试 3：节点分布统计 ======================
    std::cout << "\n===== 测试 3：节点分布统计 =====" << std::endl;
    auto distribution = answers.get_knn_node_distribution(0);
    print_knn_distribution(distribution);

    return 0;
}