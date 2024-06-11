//
//  Node.cpp
//  Strategy
//
//  Created by daiwn on 2023/6/2.
//  Copyright © 2023 Yongfeng Zhang. All rights reserved.
//

#include "Node.h"
#include <cmath>
#include "Judge.h"
#define USER 1
#define MACHINE 2
#define INF 2147483647
#define C 0.8
#define USER_WIN -1
#define MACHINE_WIN 1
#define TIE 0
#define UNFINISHED 2
using namespace std;
#include <iostream>

Node::Node(const int M, const int N, int *top, int **board, const int lastX, const int lastY, const int noX, const int noY, int currentPlayer, NodePos parent): M(M), N(N), top(top), board(board), lastX(lastX), lastY(lastY), noX(noX), noY(noY), currentPlayer(currentPlayer), parent(parent), expandableNum(0), visitedNum(0), profit(0.0) {
    expandableNodes = new int[N];
    for(int i = 0; i < N; i++) {
        if(top[i] > 0) expandableNodes[expandableNum++] = i;
    }
    children = new NodePos[N]();
}

/* 
    是否终止节点
 */
bool Node::isTerminal() {
    if(!parent || lastX == -1 || lastY == -1) return false;
    if(currentPlayer == MACHINE && userWin(lastX, lastY, M, N, board)) return true;
    if(currentPlayer == USER && machineWin(lastX, lastY, M, N, board)) return true;
    if(isTie(N, top)) return true;
    return false;
}

/* 
    拷贝棋盘
 */
int **Node::copyBoard() {
    int **newBoard = new int *[M];
    for(int i = 0; i < M; i++) {
        newBoard[i] = new int[N];
        for(int j = 0; j < N; j++) newBoard[i][j] = board[i][j];
    }
    return newBoard;
}

/* 
    销毁棋盘
 */
void Node::clearBoard(int **trashBoard) {
    for(int i = 0; i < M; i++) delete[] trashBoard[i];
    delete[] trashBoard;
}

/* 
    拷贝顶部
 */
int *Node::copyTop() {
    int *newTop = new int[N];
    for(int i = 0; i < N; i++) newTop[i] = top[i];
    return newTop;
}

/* 
    销毁数组
 */
void Node::clearArray(int *trashArray) {
    delete[] trashArray;
}


/* 
    交换执棋方
 */
int Node::opponent(int player) {
    return (player == MACHINE) ? USER : MACHINE;
}

/* 
    扩展节点
 */
NodePos Node::expand() {
    int **exBoard = copyBoard();
    int *exTop = copyTop();
    
    int randIndex = rand() % expandableNum;
    int exY = expandableNodes[randIndex], exX = exTop[exY] - 1;
    
    exTop[exY]--;
    if(exTop[exY] - 1 == noX && exY == noY) exTop[exY]--;
    exBoard[exX][exY] = currentPlayer;
    
    expandableNodes[randIndex] = expandableNodes[expandableNum-1];
    expandableNum--;
    
    return children[exY] = new Node(M, N, exTop, exBoard, exX, exY, noX, noY, opponent(currentPlayer), this);
}

/* 
    筛选信心上界最大者
 */
NodePos Node::bestChild() {
    double maxUpper = -INF;
    NodePos child, bestChild;
    for(int i = 0; i < N; i++) {
        if(!(child = children[i])) continue;
        int sign = (currentPlayer == MACHINE) ? 1 : -1;
        double uppperBound = (sign * child->profit / child->visitedNum) + C * sqrt(2 * log((double)visitedNum) / child->visitedNum);
        if(uppperBound > maxUpper) {
            maxUpper = uppperBound;
            bestChild = child;
        }
    }
    return bestChild;
}

/* 
    基于 UBC 模拟
 */
NodePos Node::treePolicy() {
    NodePos v = this;
    while(!v->isTerminal()) {
        if(v->expandableNum > 0) return v->expand();
        else v = v->bestChild();
    }
    return v;
}

/* 
    节点类型
 */
int Node::classify(int x, int y, int **boardState, int *topState, int currentPlayerState) {
    if(currentPlayerState == MACHINE && userWin(x, y, M, N, boardState)) return USER_WIN;
    if(currentPlayerState == USER && machineWin(x, y, M, N, boardState)) return MACHINE_WIN;
    if(isTie(N, topState)) return TIE;
    return UNFINISHED;
}

/* 
    随机落子
 */
int Node::randomY(int *deTop) {
    int y = rand() % N;
    while(deTop[y] <= 0) y = rand() % N;
    return y;
}

/* 
    随机模拟
 */
double Node::defaultPolicy() {
    int **deBoard = copyBoard();
    int *deTop = copyTop();
    int deX = lastX, deY = lastY;
    int deCurrentPlayer = currentPlayer;
    int deState = classify(lastX, lastY, deBoard, deTop, deCurrentPlayer);
    while(deState == UNFINISHED) {
        deY = randomY(deTop);
        
        deX = deTop[deY] - 1;
        
        deTop[deY]--;
        if(deTop[deY] - 1 == noX && deY == noY) deTop[deY]--;
        deBoard[deX][deY] = deCurrentPlayer;
        
        deCurrentPlayer = opponent(deCurrentPlayer);
        deState = classify(deX, deY, deBoard, deTop, deCurrentPlayer);
    }
    clearBoard(deBoard);
    clearArray(deTop);
    return (double)deState;
}

/* 
    反向传播
 */
void Node::backUp(double delta) {
    NodePos v = this;
    while(v) {
        v->visitedNum++;
        v->profit += delta;
        v = v->parent;
    }
}

/* 
    销毁孩子节点
 */
void Node::clearChildren() {
    for(int i = 0; i < N; i++) {
        if(children[i]) delete children[i];
    }
    delete[] children;
}

Node::~Node() {
    clearChildren();
    clearArray(expandableNodes);
    clearArray(top);
    clearBoard(board);
}
