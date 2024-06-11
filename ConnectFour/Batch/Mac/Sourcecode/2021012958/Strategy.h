/********************************************************
*	Strategy.h : 策略接口文件                           *
*	张永锋                                              *
*	zhangyf07@gmail.com                                 *
*	2014.5                                              *
*********************************************************/

#ifndef STRATEGY_H_
#define	STRATEGY_H_

#include "Point.h"

extern "C" Point* getPoint(const int M, const int N, const int* top, const int* _board, 
	const int lastX, const int lastY, const int noX, const int noY);

extern "C" void clearPoint(Point* p);

void clearArray(int M, int N, int** board);

/*
	添加你自己的辅助函数
*/

int searchMax(const int M, const int N, int **board, int *Top, int depth, int alpha, int beta, const int noX, const int noY);

int searchMin(const int M, const int N, int **board, int *Top, int depth, int alpha, int beta, const int noX, const int noY);

int evaluateBoard(const int M, const int N, int** board, const int noX, const int noY);  // 对当前局面进行评估

#endif
