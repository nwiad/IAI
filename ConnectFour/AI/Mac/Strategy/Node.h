//
//  Node.h
//  Strategy
//
//  Created by daiwn on 2023/6/1.
//  Copyright © 2023 Yongfeng Zhang. All rights reserved.
//

#ifndef Node_h
#define Node_h

using NodePos = class Node *;

class Node {
public:
    NodePos parent;  // 父节点
    NodePos *children;  // 孩子节点
    int expandableNum;  // 可扩展节点的数量
    int *expandableNodes;  // 可扩展节点的索引
    int visitedNum;  // 已访问次数
    int currentPlayer;  // 执棋方
    double profit;  // 当前收益
    int **board;  // 棋盘状态
    int *top;  // 顶部
    int M, N;  // 尺寸
    int noX, noY;  // 禁手
    int lastX, lastY;  // 上次落子的位置
    
    Node(const int M, const int N, int *top, int **board, const int lastX, const int lastY, const int noX, const int noY, int currentPlayer, NodePos parent);
    
    bool isTerminal();
    
    int **copyBoard();
    
    void clearBoard(int **trashBoard);
    
    int *copyTop();
    
    void clearArray(int *trashArray);
    
    int opponent(int player);
    
    NodePos expand();
    
    NodePos bestChild();
    
    NodePos treePolicy();
    
    int classify(int x, int y, int **boardState, int *topState, int currentPlayerState);
    
    int randomY(int *deTop);

    double defaultPolicy();
    
    void backUp(double delta);
    
    void clearChildren();
    
    ~Node();
};

#endif /* Node_h */
