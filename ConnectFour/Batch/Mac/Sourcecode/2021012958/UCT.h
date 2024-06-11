//
//  UCT.h
//  Strategy
//
//  Created by daiwn on 2023/6/1.
//  Copyright © 2023 Yongfeng Zhang. All rights reserved.
//

#ifndef UCT_h
#define UCT_h

#include "Node.h"
#include <ctime>
const double LIMIT = 1.7 * CLOCKS_PER_SEC;

class UCT {
public:
    Node *root;
    int M;
    int N;
    int noX;
    int noY;
    double start;
    
    UCT(int **board, const int *top, const int M, const int N, const int noX, const int noY): M(M), N(N), noX(noX), noY(noY) {
        int **rootBoard = new int *[M];
        int *rootTop = new int[N];
        for(int i = 0; i < M; i++) {
            rootBoard[i] = new int[N];
            for(int j = 0; j < N; j++) {
                rootBoard[i][j] = board[i][j];
            }
        }
        for(int i = 0; i < N; i++) {
            rootTop[i] = top[i];
        }
        root = new Node(rootBoard, rootTop, M, N, noX, noY);
    }
    
    Node *UCTSearch() {
        start = clock();
        while(clock() - start <= LIMIT) {
            Node *v = root->treePolicy();
            double delta = v->defaultPolicy();
            v->backUp(delta);
        }
        return root->bestChild();
    }
    
    ~UCT() {
        delete root;
    }
};

#endif /* UCT_h */
