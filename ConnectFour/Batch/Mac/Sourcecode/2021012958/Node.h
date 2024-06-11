//
//  Node.h
//  Strategy
//
//  Created by daiwn on 2023/6/1.
//  Copyright © 2023 Yongfeng Zhang. All rights reserved.
//

#ifndef Node_h
#define Node_h

#include <cmath>
#include "Judge.h"

#define EMPTY 0
#define USER 1
#define MACHINE 2
#define INF 2147483647
#define C 0.8
#define USER_WIN -1
#define MACHINE_WIN 1
#define TIE 0
#define UNFINISHED 2

class Node {
public:
    Node *parent;  // 父节点
    Node **children;  // 孩子节点
    int expandableNum;  // 可扩展节点的数量
    int *expandableNodes;  // 可扩展节点的索引
    int visitedNum = 0;  // 已访问节点的数量
    int currentPlayer;  // 执棋方
    double profit = 0.0;  // 当前收益
    int **board;  // 棋盘状态
    int *top;
    int M;
    int N;
    int noX;
    int noY;
    int lastX;  // 上次落子的位置
    int lastY;  // 上次落子的位置
    
    int x() {
        return lastX;
    }
    
    int y() {
        return lastY;
    }
    
    Node(int **board, int *top, const int M, const int N, const int noX, const int noY, int lastX = -1, int lastY = -1, int currentPlayer = MACHINE, Node *parent = nullptr): board(board), top(top), M(M), N(N), noX(noX), noY(noY), lastX(lastX), lastY(lastY), currentPlayer(currentPlayer), parent(parent) {
        children = new Node*[N]();
        expandableNodes = new int[N];
        expandableNum = 0;
        for(int y = 0; y < N; y++) {
            if(top[y] > 0) {
                expandableNodes[expandableNum++] = y;
            }
        }
    }
    
    bool isTerminal() {
        if(parent == nullptr) return false;
        if(lastX == -1 && lastY == -1) return false;
        return isTie(N, top) || (currentPlayer == MACHINE && userWin(lastX, lastY, M, N, board)) || (currentPlayer == USER && machineWin(lastX, lastY, M, N, board));
    }
    
    bool expandable(){
        return expandableNum > 0;
    }
    
    int **copyBoard() {
        int **newBoard = new int *[M];
        for(int i = 0; i < M; i++) {
            newBoard[i] = new int[N];
            for(int j = 0; j < N; j++) {
                newBoard[i][j] = board[i][j];
            }
        }
        return newBoard;
    }
    
    int *copyTop() {
        int *newTop = new int[N];
        for(int i = 0; i < N; i++) {
            newTop[i] = top[i];
        }
        return newTop;
    }
    
    Node *expand() {
        int **newBoard = copyBoard();
        int *newTop = copyTop();
        
        int randPick = rand() % expandableNum;
        int candY = expandableNodes[randPick];
        int candX = newTop[candY] - 1;
        
        newTop[candY]--;
        if(newTop[candY] - 1 == noX && candY == noY) newTop[candY]--;
        newBoard[candX][candY] = currentPlayer;
        
        expandableNum--;
        std::swap(expandableNodes[randPick], expandableNodes[expandableNum]);
        // expandableNodes[randPick] = expandableNodes[expandableNum];
        
        int current = (currentPlayer == MACHINE) ? USER : MACHINE;
        
        children[candY] = new Node(newBoard, newTop, M, N, noX, noY, candX, candY, current, this);
        return children[candY];
    }
    
    Node *bestChild() {
        double maxVal = -INF;
        Node *bestChild = nullptr;
        Node *child = nullptr;
        for(int i = 0; i < N; i++) {
            child = children[i];
            if(child == nullptr) continue;
            double comprehensive = (currentPlayer == MACHINE) ? child->profit : -child->profit;
            double uppperConfidence = (comprehensive / child->visitedNum) + C * sqrt(2 * log((double)visitedNum) / child->visitedNum);
            if(uppperConfidence > maxVal) {
                maxVal = uppperConfidence;
                bestChild = child;
            }
        }
        return bestChild;
    }
    
    Node *treePolicy() {
        Node *v = this;
        while(!v->isTerminal()) {
            if(v->expandable()) {
                return v->expand();
            }
            else {
                v = v->bestChild();
            }
        }
        return v;
    }
    
    int classify(int x, int y, int **boardState, int *topState, int currentPlayerState) {
        if(currentPlayerState == MACHINE && userWin(x, y, M, N, boardState)) return USER_WIN;
        if(currentPlayerState == USER && machineWin(x, y, M, N, boardState)) return MACHINE_WIN;
        if(isTie(N, topState)) return TIE;
        return UNFINISHED;
    }
    
    double defaultPolicy() {
        int **newBoard = copyBoard();
        int *newTop = copyTop();
        int currentPLayerState = currentPlayer;
        double profitState = classify(lastX, lastY, newBoard, newTop, currentPLayerState);
        while(profitState == UNFINISHED) {
            int candY = rand() % N;
            while(newTop[candY] <= 0) candY = rand() % N;
            int candX = newTop[candY] - 1;
            newTop[candY]--;
            if(newTop[candY] - 1 == noX && candY == noY) newTop[candY]--;
            newBoard[candX][candY] = currentPLayerState;
            
            currentPLayerState = (currentPLayerState == MACHINE) ? USER : MACHINE;
            profitState = classify(candX, candY, newBoard, newTop, currentPLayerState);
        }
        for(int i = 0; i < M; i++) {
            delete[] newBoard[i];
        }
        delete[] newBoard;
        delete[] newTop;
        return (double)profitState;
    }
    
    void backUp(double delta) {
        Node *v = this;
        while(v != nullptr) {
            v->visitedNum++;
            v->profit += delta;
            v = v->parent;
        }
    }
    
    ~Node() {
        for(int i = 0; i < N; i++) {
            if(children[i] != nullptr) delete children[i];
        }
        delete[] children;
        delete[] expandableNodes;
        for(int i = 0; i < M; i++) {
            delete[] board[i];
        }
        delete[] board;
        delete[] top;
    }
    
//    Node *contestBestChild() {
//
//    }
    
//    int pickAtRandom(int *Top) {
//        int y = rand() % N;
//        while(Top[y] <= 0) y = rand() % N;
//        return y;
//    }
    
    
    
};

#endif /* Node_h */
