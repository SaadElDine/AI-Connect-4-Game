#!/usr/bin/env python
# coding: utf-8

# # بسم الله الرحمن الرحيم

# ## Import Libraries

# In[1]:


import sys
import math
import time
import pygame
import numpy as np
import tkinter as tk
from tkinter import messagebox, simpledialog


# ## Game Engine Logic (Backend):

# ### Board Class

# In[2]:


class Board:
    
    def __init__(self):
        # Initial state of the board with the positions of the game pieces conv from binary to decimal
        self.state = 10378549747928563776 
        self.maxDepth = 1            # initial depth for minimax search 
        self.mapChildren = {}        # Stores children states
        self.mapValues = {}          # Stores Heuristic values for each state
        self.lastState= None         # Stores the state of the board from the previous move (last state)
        self.numberOfNodesExpanded=0 # Keeps track of the number of nodes expanded in minimax
        
    def getDepth(self):
        return self.maxDepth

    def setDepth(self, depth):
        self.maxDepth = depth

    def getChildrenFromMap(self, state):
        try:
            children = self.mapChildren[state]
            return children
        except:
            return None # In case a state is not found 

    def getValueFromMap(self, state):
        try:
            value = self.mapValues[state]
            return value
        except:
            return None # In case a state is not found 


# ### Usefull functions

# In[3]:


BOARD = Board() # instance of class board

def set_bit(value, bit):
    return value | (1 << bit)


def clear_bit(value, bit):
    return value & ~(1 << bit)


def getLastLocationMask(state, col): # Determines (using 3 bits) the last filled position (row) in a specified column
    return ((7 << (60 - (9 * col))) & state) >> (60 - (9 * col))
    # Masks the state using bitwise AND to extract the relevant bits then shifts the result to the right to get the value of the 3 bits at the correct position



def decimalToBinary2(n):
    return "{0:b}".format(int(n))


# Check if the state makes the board full
def isGameOver(state):
    k = 60
    for j in range(0, 7): # Iterate over columns
        maxLocation = (((7 << k) & state) >> k) # Extracts the 9 bits corresponding to the current column (Get Bit)
        if maxLocation != 7: #bin'111'
            return False
        k -= 9
    return True

# Converts the given state into a 2D array by getting the bit index for rows & col
def convertToTwoDimensions(state):
    twoDimensionalArray = np.full((6, 7), -1, np.int8)

    k = 60
    startingBits = [59, 50, 41, 32, 23, 14, 5]
    for j in range(0, 7):
        lastLocation = getLastLocationMask(state, j) - 1
        k -= 9
        for row in range(0, lastLocation):
            currentBit = ((1 << (startingBits[j] - row)) & state) >> (startingBits[j] - row) #Get Bit to index the rows and columns
            twoDimensionalArray[row][j] = currentBit
    return twoDimensionalArray # Returns a 2D array of the current state


# Converts a 2D array of the board back into the corresponding state
def convertToNumber(twoDimensionalState):
    n = 1 << 63
    k = 60
    startingBits = [59, 50, 41, 32, 23, 14, 5]
    for j in range(0, 7):
        flag = False
        for i in range(0, 6):
            if twoDimensionalState[i][j] == 1:
                n = set_bit(n, startingBits[j] - i)
            elif twoDimensionalState[i][j] == -1:
                n = (((i + 1) << k) | n)
                flag = True
                break
        if not flag:
            n = ((7 << k) | n)
        k -= 9
    return n


# ### Define First Heuristic and its used Functions:
# - Positive part:
#   - 4 consecutive AI colors gets 4 points
#   - 3 candidate consecutive (AI color) gets 3 points
#   - 2 candidate consecutive (AI color) gets 2 points
#   - stopping opponent from getting a point gets 1 point
# - Negative part
#   - 4 consecutive (Human color) gets -4 points
#   - 3 candidate consecutive (Human color) gets -3 points
#   - 2 candidate consecutive (Human color) gets -2 points
#   - stopping AI from getting a point gets -1 point

# In[4]:


# Gets the State>> C
def heuristic1(state):
    array = convertToTwoDimensions(state)
    value = 0
    for i in range(0, 6):
        for j in range(0, 7):
            if array[i][j] != -1: #checks if the cell is not empty
                value += check_neigbours1(i, j, array[i][j], array)
    return value

def check_neigbours1(x, y, value, array):
    if value == 1:
        other_player = 0
    else:
        other_player = 1
    cost = 0
    map = {}
    map[value] = 1              # Contributes positively to the cost
    map[-1] = 0                 # Empty cell >> doesn't affect the cost heur. = 0
    map[other_player] = -50     # The opponent's value contributes negatively to the cost.
    if x <= 2:
        temp = 0
        for i in range(0, 4):
            temp += map[array[x + i][y]]
        if temp > 1:
            cost += temp
        if temp == -47:
            cost -= 1

    if y <= 3:
        temp = 0
        for i in range(0, 4):
            temp += map[array[x][y + i]]
        if temp > 1:
            cost += temp
        if temp == -47:
            cost -= 1

    if y >= 3:
        temp = 0
        for i in range(0, 4):
            temp += map[array[x][y - i]]
        if temp == 3 and map[array[x][y - 3]] == 0:
            cost += 3

    if x <= 2 and y <= 3:
        temp = 0
        for i in range(0, 4):
            temp += map[array[x + i][y + i]]
        if temp > 1:
            cost += temp
        if temp == -47:
            cost -= 1

    if x <= 2 and y >= 3:
        temp = 0
        for i in range(0, 4):
            temp += map[array[x + i][y - i]]
        if temp > 1:
            cost += temp
        if temp == -47:
            cost -= 1
    if value == 1:
        return cost
    else:
        return -cost

def check_final_score1(x, y, value, array):
    if value == 1:
        other_player = 0
    else:
        other_player = 1
    cost = 0
    map = {}
    map[value] = 1
    map[-1] = 0
    map[other_player] = -50
    if x <= 2:
        temp = 0
        for i in range(0, 4):
            temp += map[array[x + i][y]]
        if temp == 4:
            cost += 4

    if y <= 3:
        temp = 0
        for i in range(0, 4):
            temp += map[array[x][y + i]]
        if temp == 4:
            cost += 4

    if x <= 2 and y <= 3:
        temp = 0
        for i in range(0, 4):
            temp += map[array[x + i][y + i]]
        if temp == 4:
            cost += 4

    if x <= 2 and y >= 3:
        temp = 0
        for i in range(0, 4):
            temp += map[array[x + i][y - i]]
        if temp == 4:
            cost += temp

    if value == 1:
        return cost
    else:
        return -cost


def get_final_score1(state):
    array = convertToTwoDimensions(state)
    value = 0
    for i in range(0, 6):
        for j in range(0, 7):
            if array[i][j] != -1:
                value += check_final_score1(i, j, array[i][j], array)
    return value


# ### Define Second Heuristic and its used Functions:
# - Positive part
#   - 4 consecutive (AI color) gets 40 points
#   - 3 candidate consecutive (AI color) gets 17 points (next move will gaurantee the point)
#   - 3 candidate consecutive (AI color) gets 15 points (a colomn is not build yet)
#   - 2 candidate consecutive (AI color) gets 4 points (next move will gaurantee the point)
#   - 2 candidate consecutive (AI color) gets 2 points (a colomn is not build yet)
#   - stopping opponent from getting a point gets 13 point
# - Negative part
#   - 4 consecutive (Human color) gets -40 points
#   - 3 candidate consecutive (Human color) gets -17 points (next move will gaurantee the point)
#   - 3 candidate consecutive (Human color) gets -15 points (a colomn is not build yet)
#   - 2 candidate consecutive (Human color) gets -4 points (next move will gaurantee the point)
#   - 2 candidate consecutive (Human color) gets -2 points (a colomn is not build yet)
#   - stopping AI from getting a point gets -13 point

# In[5]:


def heuristic2(state):
    array = convertToTwoDimensions(state)
    value = 0
    for i in range(0, 6):
        for j in range(0, 7):
            if array[i][j] != -1:
                value += check_neigbours2(i, j, array[i][j], array,state)
    return value

def check_neigbours2(x, y, value, array,state): 
    if value == 1:
        other_player = 0
    else:
        other_player = 1
    cost = 0
    map = {}
    map[value] = 1
    map[-1] = 0
    map[other_player] = -50
    last=[]
    k=60
    for i in range(0,7):
        temp = ((7<<k) & state) >> k
        last.append(temp-1)
        k-=9
    if x <= 2:
        temp = 0
        level=0
        for i in range(0, 4):
            temp += map[array[x + i][y]]
            if x + i <= last[y]:
                level += 1
        if temp == 4:
            cost += 40
        elif temp == 3 and level == 4:
            cost += 17
        elif temp == 3 and level == 3:
            cost += 15
        elif temp == 2 and level == 4:
            cost += 4
        elif temp == 2 and level < 4:
            cost += 2
        if temp == -47:
            cost -= 13

    if y <= 3:
        temp = 0
        level=0
        for i in range(0, 4):
            temp += map[array[x][y + i]]
            if x <= last[y + i]:
                level += 1
        if temp == 4:
            cost += 40
        elif temp == 3 and level == 4:
            cost += 17
        elif temp == 3 and level == 3:
            cost += 15
        elif temp == 2 and level == 4:
            cost += 4
        elif temp == 2 and level < 4:
            cost += 2
        if temp == -47:
            cost -= 13

    if y >= 3:
        temp = 0
        level=0;
        for i in range(0, 4):
                temp += map[array[x][y - i]]
                if x <= last[y - i]:
                    level += 1
        if temp == 3 and map[array[x][y - 3]]==0 and level==4:
                cost += 17
        if temp == 3 and map[array[x][y - 3]] == 0 and level < 4:
            cost += 15
        if temp == 2 and map[array[x][y]]==1 and map[array[x][y - 3]]==0 and level==4:
                cost += 4

    if x >= 3 and y <= 3:
        temp = 0
        level=0
        for i in range(0, 4):
            temp += map[array[x - i][y + i]]
            if x-i <= last[y + i]:
                level += 1
        if temp == 3 and map[array[x - 3][y + 3]] == 0 and level==4:
            cost += 17
        if temp == 3 and map[array[x - 3][y + 3]] == 0 and level==3:
            cost += 15
        if temp == 2 and map[array[x - 3][y + 3]] == 0 and level == 4:
            cost += 4
        if temp == 2 and map[array[x - 3][y + 3]] == 0 and level == 3:
            cost += 2

    if x >= 3 and y >= 3:
        temp = 0
        level=0
        for i in range(0, 4):
            temp += map[array[x - i][y - i]]
            if x-i <= last[y - i]:
                level += 1
        if temp == 3 and map[array[x - 3][y - 3]] == 0 and level == 4:
            cost += 17
        if temp == 3 and map[array[x - 3][y - 3]] == 0 and level == 3:
            cost += 15
        if temp == 2 and map[array[x - 3][y - 3]] == 0 and level == 4:
            cost += 4
        if temp == 2 and map[array[x - 3][y - 3]] == 0 and level == 3:
            cost += 2


    if x <= 2 and y <= 3:
        temp = 0
        level=0
        for i in range(0, 4):
            temp += map[array[x + i][y + i]]
            if x+i <= last[y+i]:
                level+=1
        if temp == 4:
            cost += 40
        elif temp == 3  and level==4:
            cost += 17
        elif temp == 3 and level == 3:
            cost += 15
        elif temp == 2 and level == 4:
            cost += 4
        elif temp == 2 and level < 4:
            cost += 2
        if temp == -47:
            cost -= 13

    if x <= 2 and y >= 3:
        temp = 0
        level=0
        for i in range(0, 4):
            temp += map[array[x + i][y - i]]
            if x+i <= last[y-i]:
                level += 1
        if temp == 4:
            cost += 40
        elif temp == 3 and level==4:
            cost += 17
        elif temp == 3 and level == 3:
            cost += 15
        elif temp == 2 and level ==4:
            cost += 4
        elif temp == 2 and level<4:
            cost+= 2
        if temp == -47:
            cost -= 13

    if value == 1:
        return cost
    else:
        return -cost

def check_final_score2(x, y, value, array):
    if value == 1:
        other_player = 0
    else:
        other_player = 1
    cost = 0
    map = {}
    map[value] = 1
    map[-1] = 0
    map[other_player] = -50
    if x <= 2:
        temp = 0
        for i in range(0, 4):
            temp += map[array[x + i][y]]
        if temp == 4:
            cost += 40

    if y <= 3:
        temp = 0
        for i in range(0, 4):
            temp += map[array[x][y + i]]
        if temp == 4:
            cost += 40

    if x <= 2 and y <= 3:
        temp = 0
        for i in range(0, 4):
            temp += map[array[x + i][y + i]]
        if temp == 4:
            cost += 40

    if x <= 2 and y >= 3:
        temp = 0
        for i in range(0, 4):
            temp += map[array[x + i][y - i]]
        if temp == 4:
            cost += 40
    if value == 1:
        return cost
    else:
        return -cost


def get_final_score2(state):
    array = convertToTwoDimensions(state)
    value = 0
    for i in range(0, 6):
        for j in range(0, 7):
            if array[i][j] != -1:
                value += check_final_score2(i, j, array[i][j], array)
    return value;


# In[6]:


def getChildren(player, state):
    list = [33, 42, 24, 51, 15, 60, 6]
    children = []
    for i in range(0, 7):
        k = list[i]
        temp_state = state
        temp = ((7 << k) & temp_state) >> k # Extracts the 3-bits location mask
        if player == 1 and temp != 7:
            temp_state = state | (1 << (k - temp))
            temp_state = clear_bit(temp_state, k)
            temp_state = clear_bit(temp_state, k + 1)
            temp_state = clear_bit(temp_state, k + 2)
            temp_state = temp_state | ((temp + 1) << k)
            children.append(temp_state)
        elif player == 0 and temp != 7:
            temp_state = clear_bit(temp_state, k - temp)
            temp_state = clear_bit(temp_state, k)
            temp_state = clear_bit(temp_state, k + 1)
            temp_state = clear_bit(temp_state, k + 2)
            temp_state = temp_state | ((temp + 1) << k)
            children.append(temp_state)
    return children


# ### Original minmax Algorithm:

# In[7]:


def miniMax(maxDepth, depth, isMaxPlayer, state, heuristic):
    BOARD.numberOfNodesExpanded+=1
    if depth == maxDepth:
        value = heuristic1(state) if heuristic == 0 else heuristic2(state)
        BOARD.mapValues[state] = value
        return state, value

    if isGameOver(state):
        value = get_final_score1(state) if heuristic == 0 else get_final_score2(state)
        BOARD.mapValues[state] = value
        return state, value

    children = getChildren(isMaxPlayer, state)
    BOARD.mapChildren[state] = children
    if isMaxPlayer:
        maxChild = None
        maxValue = -math.inf
        for child in children:
            childValue = miniMax(maxDepth, depth + 1, not isMaxPlayer, child, heuristic)[1]
            if childValue > maxValue:
                maxChild = child
                maxValue = childValue
        BOARD.mapValues[state] = maxValue
        return maxChild, maxValue
    else:
        minChild = None
        minValue = math.inf
        for child in children:
            childValue = miniMax(maxDepth, depth + 1, not isMaxPlayer, child, heuristic)[1]
            if childValue < minValue:
                minValue = childValue
                minChild = child
        BOARD.mapValues[state] = minValue
        return minChild, minValue


# ### minmax Algorithm with APLHA-BETA Pruning

# In[8]:


def miniMaxAlphaBeta(maxDepth, depth, isMaxPlayer, state, alpha, beta, heuristic):
    BOARD.numberOfNodesExpanded+=1
    if depth == maxDepth:
        value = heuristic1(state) if heuristic == 0 else heuristic2(state)
        BOARD.mapValues[state] = value
        return state, value

    if isGameOver(state):
        value = get_final_score1(state) if heuristic == 0 else get_final_score2(state)
        BOARD.mapValues[state] = value
        return state, value

    children = getChildren(isMaxPlayer, state)
    if isMaxPlayer:
        maxChild = None
        maxValue = -math.inf
        index = 0
        for child in children:
            childValue = miniMaxAlphaBeta(maxDepth, depth + 1, False, child, alpha, beta, heuristic)[1]
            if childValue > maxValue:
                maxChild = child
                maxValue = childValue
            if maxValue >= beta:
                break
            if maxValue > alpha:
                alpha = maxValue
            index += 1    
        for i in range(index+1,len(children)):
            children[i]= clear_bit(children[i],63)
        BOARD.mapValues[state] = maxValue
        BOARD.mapChildren[state] =children
        return maxChild, maxValue
    else:
        minChild = None
        minValue = math.inf
        index = 0
        for child in children:
            childValue = miniMaxAlphaBeta(maxDepth, depth + 1, True, child, alpha, beta, heuristic)[1]
            if childValue < minValue:
                minValue = childValue
                minChild = child
            if minValue <= alpha:
                break
            if minValue < beta:
                beta = minValue
            index += 1
        for i in range(index+1,len(children)):
            children[i]= clear_bit(children[i],63)
        BOARD.mapValues[state] = minValue
        BOARD.mapChildren[state] =children
        return minChild, minValue


# In[9]:


def nextMove(alphaBetaPruning, state, heuristic):  # The function returns the next best state in integer form
    start_time = time.time()
    BOARD.numberOfNodesExpanded=0
    BOARD.lastState= state
    if alphaBetaPruning:
        ans= miniMaxAlphaBeta(BOARD.maxDepth, 0, True, state, -math.inf, math.inf, heuristic)[0]
    else:
        ans =miniMax(BOARD.maxDepth, 0, True, state, heuristic)[0]
    # Printing the number of nodes expanded on the BOARD
    print("Number of Nodes Expanded:", BOARD.numberOfNodesExpanded)

    # Printing the elapsed time since the start_time
    print("Elapsed Time:", time.time() - start_time)
    return ans


# # FrontEnd (GUI)

# In[10]:


# Initialize a Window for the winner
win_window = tk.Tk()
# Hide this Window initialy, then show it by the end of the game using (win_window.deiconify())
win_window.withdraw()


# ### Initialize some useful variables

# In[11]:


# --------------------
# Customized Colors
# --------------------

WHITE = (255, 255, 255)
LIGHTGREY = (170, 170, 170)
GREY = (85, 85, 85)
DARKGREY = (50, 50, 50)
DARKER_GREY = (35, 35, 35)
PURPLE = (128, 0, 128)
BLACK = (0, 0, 0)
RED = (230, 30, 30)
DARKRED = (150, 0, 0)
GREEN = (30, 230, 30)
DARKGREEN = (0, 125, 0)
BLUE = (30, 30, 122)
CYAN = (30, 230, 230)
GOLD = (225, 185, 0)
DARKGOLD = (165, 125, 0)
YELLOW = (255, 255, 0)

# --------------------
# Use Defined Colors
# --------------------

BOARD_LAYOUT_BACKGROUND = BLUE
SCREEN_BACKGROUND = WHITE
FOREGROUND = WHITE
CELL_BORDER_COLOR = YELLOW
EMPTY_CELL_COLOR = WHITE

# --------------------
# Window Dimensions
# --------------------

WIDTH = 1050
HEIGHT = 742
WINDOW_SIZE = (WIDTH, HEIGHT)


# --------------------
# Board Dimensions
# --------------------

ROW_COUNT = 6
COLUMN_COUNT = 7

# --------------------
# Component Dimensions
# --------------------

SQUARE_SIZE = 100
PIECE_RADIUS = int(SQUARE_SIZE / 2 - 5)

# --------------------
# Board Coordinates
# --------------------

BOARD_BEGIN_X = 170
BOARD_BEGIN_Y = SQUARE_SIZE
BOARD_END_X = BOARD_BEGIN_X + (COLUMN_COUNT * SQUARE_SIZE)
BOARD_END_Y = BOARD_BEGIN_Y + (ROW_COUNT * SQUARE_SIZE)
BOARD_LAYOUT_END_X = BOARD_END_X + 2 * BOARD_BEGIN_X

# --------------------
# Board Dimensions
# --------------------

BOARD_WIDTH = BOARD_BEGIN_X + COLUMN_COUNT * SQUARE_SIZE
BOARD_LENGTH = ROW_COUNT * SQUARE_SIZE

# --------------------
# Player Variables
# --------------------

PIECE_COLORS = (BLUE, RED, GREEN)
PLAYER1 = 1
PLAYER2 = 2
EMPTY_CELL = 0

# --------------------
# Game-Dependent Global Variables
# --------------------

TURN = 1
GAME_OVER = False
PLAYER_SCORE = [0, 0, 0]
GAME_BOARD = [[]]
usePruning = True
screen = pygame.display.set_mode(WINDOW_SIZE)
GAME_MODE = -1
gameInSession = False
moveMade = False
HEURISTIC_USED = 1

AI_PLAYS_FIRST = False  # Set to true for AI to make the first move

nodeStack = []

# Start Game With MAX as default
minimaxCurrentMode = "MAX"


# --------------------
# Game Modes
# --------------------

SINGLE_PLAYER = 1
TWO_PLAYERS = 2     # EXTRA (BONUS)
WHO_PLAYS_FIRST = -2
MAIN_MENU = -1

# --------------------
# Developer Mode
# --------------------

# Facilitates debugging during GUI development
DEVMODE = False


# ### Initialize Game Window Class

# In[12]:


class GameWindow:
    def switch(self):
        # Transition between different states or phases of the game
        self.refreshGameWindow() # Redraw and Update Game Window
        self.gameSession()       # Start Main Game Loop 


    def setupGameWindow(self):
        global GAME_BOARD                                       # To Store Initialized Game Board 
        GAME_BOARD = self.initGameBoard(EMPTY_CELL)             # Initialize Game Board With Empty Cells (0) 
        pygame.display.set_caption('Saad & Morougue Connect 4') # Rename The Window
        self.refreshGameWindow()                                # Redraw and Update Game Window

    def refreshGameWindow(self):
        pygame.display.flip()   
        refreshBackground(LIGHTGREY, WHITE)
        self.drawGameBoard()
        self.drawGameWindowButtons()
        self.drawGameWindowLabels()

    def drawGameWindowLabels(self):
        """
        Draws all labels on the screen
        """

        if not GAME_OVER:
            captionFont = pygame.font.SysFont("Arial", 15)
            player1ScoreCaption = captionFont.render("Player1", True, WHITE)
            player2ScoreCaption = captionFont.render("Player2", True, WHITE)
            screen.blit(player1ScoreCaption, (60, 650))
            screen.blit(player2ScoreCaption, (BOARD_END_X + 65, 650))

            if GAME_MODE == SINGLE_PLAYER:
                global statsPanelY
                depthFont = pygame.font.SysFont("Serif", math.ceil(23 - len(str(BOARD.getDepth())) / 4))
                depthLabel = depthFont.render("Tree depth k = " + str(BOARD.getDepth()), True, WHITE)

                screen.blit(depthLabel, (BOARD_END_X + 5, 100))
                statsPanelY = 320

                if usePruning:
                    depthFont = pygame.font.SysFont("Arial", 15)
                    depthLabel = depthFont.render("Using ALPHA-BETA Puning", True, WHITE)
                    screen.blit(depthLabel, (BOARD_END_X + 5, 130))
                    statsPanelY += 20

        else:
            if PLAYER_SCORE[PLAYER1] == PLAYER_SCORE[PLAYER2]:
                verdict = 'IT IS A DRAW'
            elif PLAYER_SCORE[PLAYER1] > PLAYER_SCORE[PLAYER2]:
                verdict = 'Player 1 (MAX) Wins!'
            else:
                verdict = 'Player 2 (MIN) Wins!'

            verdictFont = pygame.font.SysFont("Serif", 40)
            verdictLabel = verdictFont.render(verdict, True, WHITE)
            screen.blit(verdictLabel, (430-70, 30))

        self.refreshScores()
        self.refreshStats()

    def refreshScores(self):
        #if GAME_OVER:
         #   scoreBoard_Y = BOARD_BEGIN_Y
        #else:
         #   scoreBoard_Y = 120
            
        captionFont = pygame.font.SysFont("Arial", 15)
        player1ScoreCaption = captionFont.render("Player1", True, WHITE)
        player2ScoreCaption = captionFont.render("Player2", True, WHITE)
        screen.blit(player1ScoreCaption, (60, 650))
        screen.blit(player2ScoreCaption, (BOARD_END_X + 65, 650))

        pygame.draw.rect(screen, WHITE, (25, 550, 117, 82), 0)
        player1ScoreSlot = pygame.draw.rect(screen, BOARD_LAYOUT_BACKGROUND,
                                            (25, 550, 115, 80))

        pygame.draw.rect(screen, WHITE, (BOARD_END_X + 30, 550, 117, 82), 0)
        player2ScoreSlot = pygame.draw.rect(screen, BOARD_LAYOUT_BACKGROUND,
                                            (BOARD_END_X + 30, 550, 115, 80))

        scoreFont = pygame.font.SysFont("Sans Serif", 80)
        player1ScoreCounter = scoreFont.render(str(PLAYER_SCORE[PLAYER1]), True, PIECE_COLORS[1])
        player2ScoreCounter = scoreFont.render(str(PLAYER_SCORE[PLAYER2]), True, PIECE_COLORS[2])

        player1ScoreLength = player2ScoreLength = 2.7
        if PLAYER_SCORE[PLAYER1] > 0:
            player1ScoreLength += math.log(PLAYER_SCORE[PLAYER1], 10)
        if PLAYER_SCORE[PLAYER2] > 0:
            player2ScoreLength += math.log(PLAYER_SCORE[PLAYER2], 10)

        screen.blit(player1ScoreCounter,
                    (3 + player1ScoreSlot.x + player1ScoreSlot.width / player1ScoreLength, 563))
        screen.blit(player2ScoreCounter,
                    (6 + player2ScoreSlot.x + player2ScoreSlot.width / player2ScoreLength, 563))

    def mouseOverMainLabel(self):  # Check If Mouse is on the game board
        return 20 <= pygame.mouse.get_pos()[1] <= 45 and 810 <= pygame.mouse.get_pos()[0] <= 1030
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def refreshStats(self):
        """
        Refreshes the analysis section
        """
        global statsPanelY
        if GAME_MODE == SINGLE_PLAYER:
            if GAME_OVER:
                statsPanelY = showStatsButton.y + showStatsButton.height + 5
            pygame.draw.rect(
                screen, WHITE,
                (BOARD_LAYOUT_END_X + 9, statsPanelY + 5, WIDTH - BOARD_LAYOUT_END_X - 18, 267 + (370 - statsPanelY)),
                0)
            pygame.draw.rect(
                screen, GREY,
                (BOARD_LAYOUT_END_X + 10, statsPanelY + 6, WIDTH - BOARD_LAYOUT_END_X - 20, 265 + (370 - statsPanelY)))

    ######   Buttons    ######

    def drawGameWindowButtons(self):
        """
            Draws all buttons on the screen
            """
        global showStatsButton, contributorsButton, playAgainButton, settingsButton, homeButton
        global settingsIcon, settingsIconAccent, homeIcon, homeIconAccent

        settingsIcon = pygame.image.load('settings-icon.png').convert_alpha()
        settingsIconAccent = pygame.image.load('settings-icon-accent.png').convert_alpha()
        homeIcon = pygame.image.load('home-icon.png').convert_alpha()
        homeIconAccent = pygame.image.load('home-icon-accent.png').convert_alpha()

        contributorsButton = Button(
            screen, color=BLUE,
            x=BOARD_END_X + 80, y=680,
            width=80, height=40, text="BY")
        contributorsButton.draw(WHITE)

        settingsButton = Button(window=screen, color= LIGHTGREY, x=WIDTH - 48, y=BOARD_BEGIN_Y - 70,
                                width=35, height=35)

        homeButton = Button(window=screen, color= LIGHTGREY, x=WIDTH - 88, y=BOARD_BEGIN_Y - 70,
                            width=35, height=35)
        self.reloadSettingsButton(settingsIcon)
        self.reloadHomeButton(homeIcon)

        if GAME_OVER:

            playAgainButton = Button(
                window=screen, color=BLUE, x=20, y=100,
                width=130, height=82, text="NEW GAME")
            playAgainButton.draw()

        if GAME_MODE == SINGLE_PLAYER:
            if moveMade:
                statsButtonColor = BLUE
            else:
                statsButtonColor = GREY
                
            showStatsButton = Button(window=screen, color=statsButtonColor, x=BOARD_END_X + 30, y=450, width=117, height=82, text=" SHOW TREE")
            showStatsButton.draw(WHITE)

    def reloadSettingsButton(self, icon):
        settingsButton.draw()
        screen.blit(icon, (settingsButton.x + 2, settingsButton.y + 2))

    def reloadHomeButton(self, icon):
        homeButton.draw()
        screen.blit(icon, (homeButton.x + 2, homeButton.y + 2))

    ######   Game Board  ######

    def initGameBoard(self, initialCellValue):
        """
        Initializes the game board with the value given.
        :param initialCellValue: Value of initial cell value
        :return: board list with all cells initialized to initialCellValue
        """
        global GAME_BOARD
        GAME_BOARD = np.full((ROW_COUNT, COLUMN_COUNT), initialCellValue)
        return GAME_BOARD

    def printGameBoard(self):
        """
        Prints the game board to the terminal
        """
        print('\n-\n' +
              str(GAME_BOARD) +
              '\n Player ' + str(TURN) + ' plays next')

    def drawGameBoard(self):
        """
        Draws the game board on the interface with the latest values in the board list
        """
        pygame.draw.rect(screen, BLACK, (0, 0, BOARD_LAYOUT_END_X, HEIGHT), 0)
        boardLayout = pygame.draw.rect(
            screen, BOARD_LAYOUT_BACKGROUND, (0, 0, BOARD_LAYOUT_END_X - 1, HEIGHT))
        gradientRect(screen, GREY, LIGHTGREY, boardLayout)
        for c in range(COLUMN_COUNT):
            for r in range(ROW_COUNT):
                col = BOARD_BEGIN_X + (c * SQUARE_SIZE)
                row = BOARD_BEGIN_Y + (r * SQUARE_SIZE)
                piece = GAME_BOARD[r][c]
                pygame.draw.rect(
                    screen, CELL_BORDER_COLOR, (col, row, SQUARE_SIZE, SQUARE_SIZE))
                pygame.draw.circle(
                    screen, PIECE_COLORS[piece], (int(col + SQUARE_SIZE / 2), int(row + SQUARE_SIZE / 2)), PIECE_RADIUS)
        pygame.display.update()

    def hoverPieceOverSlot(self):
        """
        Hovers the piece over the game board with the corresponding player's piece color
        """
        boardLayout = pygame.draw.rect(screen, BOARD_LAYOUT_BACKGROUND,
                                       (0, BOARD_BEGIN_Y - SQUARE_SIZE, BOARD_WIDTH + SQUARE_SIZE / 2, SQUARE_SIZE))
        gradientRect(screen, GREY, LIGHTGREY, boardLayout)
        posx = pygame.mouse.get_pos()[0]
        if BOARD_BEGIN_X < posx < BOARD_END_X:
            pygame.mouse.set_visible(False)
            pygame.draw.circle(screen, PIECE_COLORS[TURN], (posx, int(SQUARE_SIZE / 2)), PIECE_RADIUS)
        else:
            pygame.mouse.set_visible(True)

    def dropPiece(self, col, piece) -> tuple:
        """
        Drops the given piece in the next available cell in slot 'col'
        :param col: Column index where the piece will be dropped
        :param piece: Value of the piece to be put in array.
        :returns: tuple containing the row and column of piece position
        """
        row = self.getNextOpenRow(col)
        GAME_BOARD[row][col] = piece

        return row, col

    def hasEmptyCell(self, col) -> bool:
        """
        Checks if current slot has an empty cell. Assumes col is within array limits
        :param col: Column index representing the slot
        :return: True if slot has an empty cell. False otherwise.
        """
        return GAME_BOARD[0][col] == EMPTY_CELL

    def getNextOpenRow(self, col):
        """
        Gets the next available cell in the slot
        :param col: Column index
        :return: If exists, the row of the first available empty cell in the slot. None otherwise.
        """
        for r in range(ROW_COUNT - 1, -1, -1):
            if GAME_BOARD[r][col] == EMPTY_CELL:
                return r
        return None

    def boardIsFull(self) -> bool:
        """
        Checks if the board game is full
        :return: True if the board list has no empty slots, False otherwise.
        """
        for slot in range(COLUMN_COUNT):
            if self.hasEmptyCell(slot):
                return False
        return True

    def getBoardColumnFromPos(self, posx):
        """
        Get the index of the board column corresponding to the given position
        :param posx: Position in pixels
        :return: If within board bounds, the index of corresponding column, None otherwise
        """
        column = int(math.floor(posx / SQUARE_SIZE))
        if 0 <= column < COLUMN_COUNT:
            return column
        return None

    def buttonResponseToMouseEvent(self, event):
        """
        Handles button behaviour in response to mouse events influencing them
        """
        if event.type == pygame.MOUSEMOTION:
            if GAME_MODE == SINGLE_PLAYER and showStatsButton.isOver(event.pos):
                if moveMade and not GAME_OVER:
                    pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
                    alterButtonAppearance(showStatsButton, YELLOW, WHITE)
            elif contributorsButton.isOver(event.pos):
                pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
                alterButtonAppearance(contributorsButton, YELLOW, WHITE)
            elif settingsButton.isOver(event.pos):
                pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
                self.reloadSettingsButton(settingsIconAccent)
            elif homeButton.isOver(event.pos):
                pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
                self.reloadHomeButton(homeIconAccent)
            elif GAME_OVER and playAgainButton.isOver(event.pos):
                pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
                alterButtonAppearance(
                    button=playAgainButton, color=GREEN, outlineColor=WHITE, hasGradBackground=True,
                    gradLeftColor=WHITE, gradRightColor=GREEN, fontSize=18)
            elif self.mouseOverMainLabel():
                pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
            else:
                pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)
                if GAME_MODE == SINGLE_PLAYER:
                    if moveMade and not GAME_OVER:
                        alterButtonAppearance(showStatsButton, BLUE, WHITE)
                    else:
                        alterButtonAppearance(showStatsButton, GREY, WHITE)
                alterButtonAppearance(contributorsButton, BLUE, WHITE)
                self.reloadSettingsButton(settingsIcon)
                self.reloadHomeButton(homeIcon)
                if GAME_OVER:
                    alterButtonAppearance(playAgainButton, GREEN, BLACK)

        if event.type == pygame.MOUSEBUTTONDOWN:
            if GAME_MODE == SINGLE_PLAYER and not GAME_OVER and showStatsButton.isOver(event.pos) and moveMade:
                alterButtonAppearance(showStatsButton, GOLD, WHITE)
            elif contributorsButton.isOver(event.pos):
                alterButtonAppearance(contributorsButton, RED, WHITE)
            elif GAME_OVER and playAgainButton.isOver(event.pos):
                alterButtonAppearance(
                    button=playAgainButton, color=GREEN, outlineColor=BLACK, hasGradBackground=True,
                    gradLeftColor=GREEN, gradRightColor=WHITE)
            elif self.mouseOverMainLabel() or homeButton.isOver(event.pos):
                self.resetEverything()
                mainMenu.setupMainMenu()
                mainMenu.show()
            elif settingsButton.isOver(event.pos):
                settingsWindow = SettingsWindow()
                settingsWindow.setupSettingsMenu()
                settingsWindow.show()

        if event.type == pygame.MOUSEBUTTONUP:
            if GAME_MODE == SINGLE_PLAYER and not GAME_OVER and showStatsButton.isOver(event.pos) and moveMade:
                alterButtonAppearance(showStatsButton, BLUE, WHITE)
                treevisualizer = TreeVisualizer()
                treevisualizer.switch()
            elif contributorsButton.isOver(event.pos):
                alterButtonAppearance(contributorsButton, BLUE, WHITE)
                self.showContributors()
            elif GAME_OVER and playAgainButton.isOver(event.pos):
                alterButtonAppearance(
                    button=playAgainButton, color=GREEN, outlineColor=WHITE, hasGradBackground=True,
                    gradLeftColor=WHITE, gradRightColor=GREEN, fontSize=22)
                self.resetEverything()

        if DEVMODE:
            pygame.draw.rect(screen, BLACK, (BOARD_LAYOUT_END_X + 20, 70, WIDTH - BOARD_LAYOUT_END_X - 40, 40))
            pygame.mouse.set_visible(True)
            titleFont = pygame.font.SysFont("Sans Serif", 20, False, True)
            coordinates = titleFont.render(str(pygame.mouse.get_pos()), True, WHITE)
            screen.blit(coordinates, (BOARD_LAYOUT_END_X + 100, 80))

    def showContributors(self):
        """
        Invoked at pressing the contributors button. Displays a message box Containing names and IDs of contributors
        """
        messagebox.showinfo('Contributors', "7370   -   Saad El Dine Ahmed\n"
                                            "7524   -   Morougue Mahmoud Ghazal\n")

    def gameSession(self):
        """
        Runs the game session
        """
        global GAME_OVER, TURN, GAME_BOARD, gameInSession, moveMade, AI_PLAYS_FIRST
        gameInSession = True
        nodeStack.clear()

        while True:

            if AI_PLAYS_FIRST and not moveMade:
                switchTurn()
                self.player2Play()
                moveMade = True

            pygame.display.update()

            if not GAME_OVER:
                self.hoverPieceOverSlot()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()

                self.buttonResponseToMouseEvent(event)

                if not GAME_OVER and event.type == pygame.MOUSEBUTTONDOWN:
                    posx = event.pos[0] - BOARD_BEGIN_X
                    column = self.getBoardColumnFromPos(posx)

                    if column is not None:
                        if self.hasEmptyCell(column):
                            self.dropPiece(column, TURN)
                            self.computeScore()
                            switchTurn()
                            self.refreshGameWindow()

                            moveMade = True

                            if not self.boardIsFull():
                                self.player2Play()

                            if self.boardIsFull():
                                GAME_OVER = True
                                pygame.mouse.set_visible(True)
                                self.refreshGameWindow()
                                break

    def player2Play(self):
        if GAME_MODE == SINGLE_PLAYER:
            self.computerPlay()
        elif GAME_MODE == TWO_PLAYERS:
            pass
        
#  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    def computerPlay(self):
        global GAME_BOARD, parentState
        for i in range(ROW_COUNT):
            for j in range(COLUMN_COUNT):
                GAME_BOARD[i][j] -= 1
   
        flippedGameBoard = np.flip(m=GAME_BOARD, axis=0)  # Flip about x-axis
        numericState = convertToNumber(flippedGameBoard)
        boardState = nextMove(alphaBetaPruning=usePruning, state=numericState, heuristic=HEURISTIC_USED)
        flippedNewState = convertToTwoDimensions(boardState)
        newState = np.flip(m=flippedNewState, axis=0)  # Flip about x-axis

        for i in range(ROW_COUNT):
            for j in range(COLUMN_COUNT):
                GAME_BOARD[i][j] += 1
                newState[i][j] += 1

        newC = self.getNewMove(newState=newState, oldState=GAME_BOARD)

        boardLayout = pygame.draw.rect(screen, BOARD_LAYOUT_BACKGROUND,
                                       (0, BOARD_BEGIN_Y - SQUARE_SIZE, BOARD_WIDTH + SQUARE_SIZE / 2, SQUARE_SIZE))
        for i in range(BOARD_BEGIN_X, math.ceil(BOARD_BEGIN_X + newC * SQUARE_SIZE + SQUARE_SIZE / 2), 2):
            gradientRect(screen, GREY, LIGHTGREY, boardLayout)
            pygame.draw.circle(
                screen, PIECE_COLORS[TURN], (i, int(SQUARE_SIZE / 2)), PIECE_RADIUS)
            pygame.display.update()
        self.refreshGameWindow()

        self.hoverPieceOverSlot()

        GAME_BOARD = newState
        self.computeScore()

        switchTurn()
        self.refreshGameWindow()

    def resetEverything(self):
        """
        Resets everything back to default values
        """
        global GAME_BOARD, PLAYER_SCORE, GAME_OVER, TURN, moveMade
        PLAYER_SCORE = [0, 0, 0]
        GAME_OVER = False
        TURN = 1
        moveMade = False
        self.setupGameWindow()

    def getNewMove(self, newState, oldState) -> int:
        """
        :return: New move made by the AI
        """
        for r in range(ROW_COUNT):
            for c in range(COLUMN_COUNT):
                if newState[r][c] != oldState[r][c]:
                    return c

    def computeScore(self):
        """
        Computes every player's score and stores it in the global PLAYER_SCORES list
        :returns: values in PLAYER_SCORES list
        """
        global PLAYER_SCORE
        PLAYER_SCORE = [0, 0, 0]
        for r in range(ROW_COUNT):
            consecutive = 0
            for c in range(COLUMN_COUNT):
                consecutive += 1
                if c > 0 and GAME_BOARD[r][c] != GAME_BOARD[r][c - 1]:
                    consecutive = 1
                if consecutive >= 4:
                    PLAYER_SCORE[GAME_BOARD[r][c]] += 1

        for c in range(COLUMN_COUNT):
            consecutive = 0
            for r in range(ROW_COUNT):
                consecutive += 1
                if r > 0 and GAME_BOARD[r][c] != GAME_BOARD[r - 1][c]:
                    consecutive = 1
                if consecutive >= 4:
                    PLAYER_SCORE[GAME_BOARD[r][c]] += 1

        for r in range(ROW_COUNT - 3):
            for c in range(COLUMN_COUNT - 3):
                if GAME_BOARD[r][c] == GAME_BOARD[r + 1][c + 1]                         == GAME_BOARD[r + 2][c + 2] == GAME_BOARD[r + 3][c + 3]:
                    PLAYER_SCORE[GAME_BOARD[r][c]] += 1

        for r in range(ROW_COUNT - 3):
            for c in range(COLUMN_COUNT - 1, 2, -1):
                if GAME_BOARD[r][c] == GAME_BOARD[r + 1][c - 1]                         == GAME_BOARD[r + 2][c - 2] == GAME_BOARD[r + 3][c - 3]:
                    PLAYER_SCORE[GAME_BOARD[r][c]] += 1

        return PLAYER_SCORE

    def isWithinBounds(self, mat, r, c) -> bool:
        """
        :param mat: 2D matrix to check in
        :param r: current row
        :param c: current column
        :return: True if coordinates are within matrix bounds, False otherwise
        """
        return 0 <= r <= len(mat) and 0 <= c <= len(mat[0])


# ### Main Menu Class

# In[13]:


class MainMenu:
    def switch(self):
        self.setupMainMenu()
        self.show()

    def show(self):
        while GAME_MODE == MAIN_MENU:
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()

                self.buttonResponseToMouseEvent(event)

        if GAME_MODE == WHO_PLAYS_FIRST:
            WhoPlaysFirstMenu().switch()
        else:
            startGameSession()

    def setupMainMenu(self):
        """
        Initializes the all components in the frame
        """
        global GAME_MODE, gameInSession
        GAME_MODE = MAIN_MENU
        gameInSession = False
        pygame.display.flip()
        pygame.display.set_caption('Saad & Morougue Connect 4 - Main Menu')
        self.refreshMainMenu()

    def refreshMainMenu(self):
        """
        Refreshes the screen and all the components
        """
        pygame.display.flip()

        # Draw Background Image
        background_image = pygame.image.load("background.png")  
        screen.blit(background_image, (0, 0))  # Blit the image onto the screen at the specified position

        # Draw Buttons and Labels
        self.drawMainMenuButtons()
        self.drawMainMenuLabels()


    def drawMainMenuButtons(self):
        global singlePlayerButton, multiPlayerButton, settingsButton
        global settingsIcon, settingsIconAccent

        settingsIcon = pygame.image.load('settings-icon.png').convert_alpha()
        settingsIconAccent = pygame.image.load('settings-icon-accent.png').convert_alpha()
        
        singlePlayerButton = Button(
            window=screen, color=LIGHTGREY, x=WIDTH / 3 - 165, y=HEIGHT / 3 - 126, width=WIDTH / 5, height=HEIGHT / 5,
            gradCore=True, coreLeftColor=RED, coreRightColor=WHITE, text='PLAY AGAINST AI')

        multiPlayerButton = Button(
            window=screen, color=LIGHTGREY, x=WIDTH / 3 + 320, y=HEIGHT / 3 + HEIGHT / 5 + 90, width=WIDTH / 5,
            height=HEIGHT / 5, gradCore=True, coreLeftColor=YELLOW, coreRightColor=WHITE, text='TWO-PLAYERS')
        
        settingsButton = Button(window=screen, color=(82, 82, 82), x=WIDTH / 3 - 70, y=HEIGHT / 3 + HEIGHT / 5  + 160, width=35, height=35, shape='ellipse')
        
        self.reloadSettingsButton(settingsButton, settingsIcon)
        singlePlayerButton.draw(BLACK, 2)
        multiPlayerButton.draw(BLACK, 2)
    
    def drawMainMenuLabels(self):
        titleFont = pygame.font.SysFont("", 65, True, True)
        mainLabel = titleFont.render("", True, WHITE)
        screen.blit(mainLabel, (WIDTH / 5, HEIGHT / 8))

    def reloadSettingsButton(self, button, icon):
        button.draw()
        screen.blit(icon, (button.x + 2, button.y + 2))

    def buttonResponseToMouseEvent(self, event):
        """
        Handles button behavior in response to mouse events influencing them
        """
        try:
            if event.type == pygame.MOUSEMOTION:
                if singlePlayerButton.isOver(event.pos):
                    pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
                    alterButtonAppearance(singlePlayerButton, WHITE, BLACK,
                                          hasGradBackground=True, gradLeftColor=WHITE, gradRightColor=RED)
                elif multiPlayerButton.isOver(event.pos):
                    pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
                    alterButtonAppearance(multiPlayerButton, WHITE, BLACK,
                                          hasGradBackground=True, gradLeftColor=WHITE, gradRightColor=YELLOW)
                elif settingsButton.isOver(event.pos):
                    pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
                    self.reloadSettingsButton(settingsButton, settingsIconAccent)
                else:
                    pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)
                    alterButtonAppearance(singlePlayerButton, LIGHTGREY, BLACK,
                                          hasGradBackground=True, gradLeftColor=RED, gradRightColor=WHITE)
                    alterButtonAppearance(multiPlayerButton, LIGHTGREY, BLACK,
                                          hasGradBackground=True, gradLeftColor=YELLOW, gradRightColor=WHITE)
                    self.reloadSettingsButton(settingsButton, settingsIcon)

            if event.type == pygame.MOUSEBUTTONDOWN:
                if singlePlayerButton.isOver(event.pos):
                    alterButtonAppearance(singlePlayerButton, WHITE, BLACK,
                                          hasGradBackground=True, gradLeftColor=GOLD, gradRightColor=RED)
                elif multiPlayerButton.isOver(event.pos):
                    alterButtonAppearance(multiPlayerButton, WHITE, BLACK,
                                          hasGradBackground=True, gradLeftColor=GOLD, gradRightColor=YELLOW)
                elif settingsButton.isOver(event.pos):
                    settingsWindow = SettingsWindow()
                    settingsWindow.setupSettingsMenu()
                    settingsWindow.show()

            if event.type == pygame.MOUSEBUTTONUP:
                global GAME_MODE
                if singlePlayerButton.isOver(event.pos):
                    alterButtonAppearance(singlePlayerButton, WHITE, BLACK,
                                          hasGradBackground=True, gradLeftColor=RED, gradRightColor=WHITE)
                    setGameMode(WHO_PLAYS_FIRST)
                elif multiPlayerButton.isOver(event.pos):
                    alterButtonAppearance(multiPlayerButton, WHITE, BLACK,
                                          hasGradBackground=True, gradLeftColor=YELLOW, gradRightColor=WHITE)
                    setGameMode(TWO_PLAYERS)
                elif settingsButton.isOver(event.pos):
                    alterButtonAppearance(multiPlayerButton, WHITE, BLACK,
                                          hasGradBackground=True, gradLeftColor=YELLOW, gradRightColor=RED)
                    settingsWindow = SettingsWindow()
                    settingsWindow.switch()
        except SystemExit:
            exit()


# ### Initializer Agent Class

# In[14]:


class WhoPlaysFirstMenu:
    def switch(self):
        self.setupWPFMenu()
        self.show()

    def show(self):
        while GAME_MODE == WHO_PLAYS_FIRST:
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()

                self.buttonResponseToMouseEvent(event)

        startGameSession()

    def setupWPFMenu(self):
        """
        Initializes the all components in the frame
        """
        pygame.display.flip()
        pygame.display.set_caption('Saad & Morougue Connect 4 - Who Plays First?')
        self.refreshWPFMenu()

    def refreshWPFMenu(self):
        """
        Refreshes the screen and all the components
        """
        pygame.display.flip()
        # Draw Background Image
        background_image = pygame.image.load("backgroud.png")  
        screen.blit(background_image, (0, 0))  # Blit the image onto the screen at the specified position
        self.drawWPFButtons()
        self.drawWPFLabels()

    def reloadBackButton(self, icon):
        backButton.draw()
        screen.blit(icon, (backButton.x + 2, backButton.y + 2))

    def drawWPFButtons(self):
        global playerFirstButton, computerFirstButton
        global backButton, backIcon, backIconAccent

        backIconAccent = pygame.image.load('back-icon.png').convert_alpha()
        backIcon = pygame.image.load('back-icon-accent.png').convert_alpha()

        backButton = Button(window=screen, color=(81, 81, 81), x=30, y=680, width=52, height=52)
        self.reloadBackButton(backIcon)

        playerFirstButton = Button(
            window=screen, color=YELLOW, x=WIDTH / 2 - 370, y=HEIGHT / 2, width=300, height=HEIGHT / 5,
            gradCore=True, coreLeftColor=RED, coreRightColor=BLACK, text='HUMAN')

        computerFirstButton = Button(
            window=screen, color=YELLOW, x=WIDTH / 2 + 100, y=HEIGHT / 2, width=300,
            height=HEIGHT / 5,
            gradCore=True, coreLeftColor=BLACK, coreRightColor=RED, text='COMPUTER')

        playerFirstButton.draw(BLACK, 2)
        computerFirstButton.draw(BLACK, 2)

    def drawWPFLabels(self):
        titleFont = pygame.font.SysFont("Sans Serif", 65, True, True)
        mainLabel = titleFont.render("Which Player Will Start?", True, BLACK)
        screen.blit(mainLabel, (WIDTH / 2 - mainLabel.get_width() / 2, -50 + (HEIGHT / 3 - mainLabel.get_height() / 2)))

    def buttonResponseToMouseEvent(self, event):
        """
        Handles button behaviour in response to mouse events influencing them
        """
        if event.type == pygame.MOUSEMOTION:
            if playerFirstButton.isOver(event.pos):
                pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
                alterButtonAppearance(playerFirstButton, WHITE, BLACK,
                                      hasGradBackground=True, gradLeftColor=BLACK, gradRightColor=RED)
            elif computerFirstButton.isOver(event.pos):
                pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
                alterButtonAppearance(computerFirstButton, WHITE, BLACK,
                                      hasGradBackground=True, gradLeftColor=RED, gradRightColor=BLACK)
            elif backButton.isOver(event.pos):
                pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
                self.reloadBackButton(backIconAccent)
            else:
                pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)
                alterButtonAppearance(playerFirstButton, LIGHTGREY, BLACK,
                                      hasGradBackground=True, gradLeftColor=RED, gradRightColor=BLACK)
                alterButtonAppearance(computerFirstButton, LIGHTGREY, BLACK,
                                      hasGradBackground=True, gradLeftColor=BLACK, gradRightColor=RED)
                self.reloadBackButton(backIcon)

        if event.type == pygame.MOUSEBUTTONDOWN:
            if playerFirstButton.isOver(event.pos):
                alterButtonAppearance(playerFirstButton, WHITE, BLACK,
                                      hasGradBackground=True, gradLeftColor=GOLD, gradRightColor=RED)
            elif computerFirstButton.isOver(event.pos):
                alterButtonAppearance(computerFirstButton, WHITE, BLACK,
                                      hasGradBackground=True, gradLeftColor=GOLD, gradRightColor=BLACK)
            elif backButton.isOver(event.pos):
                MainMenu().switch()

        if event.type == pygame.MOUSEBUTTONUP:
            global GAME_MODE, AI_PLAYS_FIRST
            if playerFirstButton.isOver(event.pos):
                alterButtonAppearance(playerFirstButton, WHITE, BLACK,
                                      hasGradBackground=True, gradLeftColor=RED, gradRightColor=BLACK)
                AI_PLAYS_FIRST = False
                setGameMode(SINGLE_PLAYER)
            elif computerFirstButton.isOver(event.pos):
                alterButtonAppearance(computerFirstButton, WHITE, BLACK,
                                      hasGradBackground=True, gradLeftColor=BLACK, gradRightColor=RED)
                AI_PLAYS_FIRST = True
                setGameMode(SINGLE_PLAYER)


# ### Bonus Part ( Visualize Tree )

# In[15]:


class TreeVisualizer:
    def switch(self):
        self.setupTreeVisualizer()
        self.show()

    def show(self):
        while True:
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()

                self.buttonResponseToMouseEvent(event)

    def setupTreeVisualizer(self):
        global minimaxCurrentMode
        minimaxCurrentMode = 'MAX'
        pygame.display.flip()
        pygame.display.set_caption('Saad & Morougue - Tree Visualizer')
        self.refreshTreeVisualizer(rootNode=None)

    def refreshTreeVisualizer(self, rootNode=None):
        refreshBackground(LIGHTGREY, WHITE)
        self.drawTreeNodes(rootNode)
        self.drawTreeVisualizerButtons()
        self.drawTreeVisualizerLabels()
        self.drawMiniGameBoard()
        pygame.display.update()

    def drawTreeNodes(self, parent):
        global parentNodeButton, rootNodeButton, child1Button, child2Button, child3Button, child4Button, child5Button, child6Button, child7Button
        global root, child1, child2, child3, child4, child5, child6, child7
        child1 = child2 = child3 = child4 = child5 = child6 = child7 = None

        parentNodeButton = Button(window=screen, color=YELLOW, x=WIDTH / 2 - 70, y=10, width=140, height=100,
                                  text='BACK TO PARENT',
                                  shape='ellipse')
        parentNodeButton.draw(BLACK)

        if parent is None:
            root = BOARD.lastState
            nodeStack.append(root)
            rootValue = BOARD.getValueFromMap(BOARD.lastState)
        else:
            root = nodeStack[-1]
            rootValue = BOARD.getValueFromMap(root)

        rootNodeButton = Button(window=screen, color=PURPLE, x=WIDTH / 2 - 70, y=parentNodeButton.y + 200, width=140,
                                height=100, text=str(rootValue),
                                shape='ellipse')

        children = BOARD.getChildrenFromMap(root)

        color, txt = GREY, ''
        if children is not None and len(children) >= 1:
            child1 = children[0]
            color, txt = self.styleNode(child1)
        child1Button = Button(window=screen, color=color, x=40, y=rootNodeButton.y + 300, width=140, height=100,
                              text=txt, shape='ellipse')

        color, txt = GREY, ''
        if children is not None and len(children) >= 2:
            child2 = children[1]
            color, txt = self.styleNode(child2)
        child2Button = Button(window=screen, color=color, x=180, y=rootNodeButton.y + 200, width=140, height=100,
                              text=txt, shape='ellipse')

        color, txt = GREY, ''
        if children is not None and len(children) >= 3:
            child3 = children[2]
            color, txt = self.styleNode(child3)
        child3Button = Button(window=screen, color=color, x=320, y=rootNodeButton.y + 300, width=140, height=100,
                              text=txt, shape='ellipse')

        color, txt = GREY, ''
        if children is not None and len(children) >= 4:
            child4 = children[3]
            color, txt = self.styleNode(child4)
        child4Button = Button(window=screen, color=color, x=460, y=rootNodeButton.y + 200, width=140, height=100,
                              text=txt, shape='ellipse')

        color, txt = GREY, ''
        if children is not None and len(children) >= 5:
            child5 = children[4]
            color, txt = self.styleNode(child5)
        child5Button = Button(window=screen, color=color, x=600, y=rootNodeButton.y + 300, width=140, height=100,
                              text=txt, shape='ellipse')

        color, txt = GREY, ''
        if children is not None and len(children) >= 6:
            child6 = children[5]
            color, txt = self.styleNode(child6)
        child6Button = Button(window=screen, color=color, x=740, y=rootNodeButton.y + 200, width=140, height=100,
                              text=txt, shape='ellipse')

        color, txt = GREY, ''
        if children is not None and len(children) >= 7:
            child7 = children[6]
            color, txt = self.styleNode(child7)
        child7Button = Button(window=screen, color=color, x=880, y=rootNodeButton.y + 300, width=140, height=100,
                              text=txt, shape='ellipse')

        pygame.draw.rect(screen, BLACK, (
            rootNodeButton.x + rootNodeButton.width / 2, rootNodeButton.y + rootNodeButton.height + 10, 2, 80))
        pygame.draw.rect(screen, BLACK,
                         (
                             rootNodeButton.x + rootNodeButton.width / 2,
                             parentNodeButton.y + parentNodeButton.height + 10,
                             2, 80))
        horizontalRule = pygame.draw.rect(screen, BLACK,
                                          (child1Button.x + child1Button.width / 2,
                                           rootNodeButton.y + rootNodeButton.height + 50,
                                           WIDTH - (child1Button.x + child1Button.width / 2)
                                           - (WIDTH - (child7Button.x + child7Button.width / 2)), 2))
        pygame.draw.rect(screen, BLACK, (child2Button.x + child2Button.width / 2, horizontalRule.y, 2, 40))
        pygame.draw.rect(screen, BLACK, (child6Button.x + child6Button.width / 2, horizontalRule.y, 2, 40))
        pygame.draw.rect(screen, BLACK,
                         (child1Button.x + child1Button.width / 2, horizontalRule.y, 2, 40 + child2Button.height))
        pygame.draw.rect(screen, BLACK,
                         (child7Button.x + child7Button.width / 2, horizontalRule.y, 2, 40 + child2Button.height))
        pygame.draw.rect(screen, BLACK,
                         (child3Button.x + child3Button.width / 2, horizontalRule.y, 2, 40 + child2Button.height))
        pygame.draw.rect(screen, BLACK,
                         (child5Button.x + child5Button.width / 2, horizontalRule.y, 2, 40 + child2Button.height))

        rootNodeButton.draw()
        child1Button.draw()
        child2Button.draw()
        child3Button.draw()
        child4Button.draw()
        child5Button.draw()
        child6Button.draw()
        child7Button.draw()

    def styleNode(self, state):
        if self.isNull(state):
            return GREY, ''
        if self.isPruned(state):
            return DARKRED, 'PRUNED'
        value = BOARD.getValueFromMap(state)
        return BLUE, str(value)

    def navigateNode(self, node, rootNode, nodeButton):
        global root
        if node is not None and BOARD.getChildrenFromMap(node) is not None:
            nodeStack.append(node)
            self.toggleMinimaxCurrentMode()

            rootY, nodeY = rootNodeButton.y, nodeButton.y
            rootX, nodeX = rootNodeButton.x, nodeButton.x
            while nodeX not in range(int(rootX) - 3, int(rootX) + 3)                     or nodeY not in range(int(rootY) - 3, int(rootY) + 3):
                if nodeX < rootX and nodeX not in range(int(rootX) - 3, int(rootX) + 3):
                    nodeX += 2
                elif nodeX > rootX and nodeX not in range(int(rootX) - 3, int(rootX) + 3):
                    nodeX -= 2
                if nodeY > rootY and nodeY not in range(int(rootY) - 3, int(rootY) + 3):
                    nodeY -= 2
                color = BLUE
                if math.sqrt(pow(rootX - nodeX, 2) + pow(rootY - nodeY, 2)) <= 200:
                    color = LIGHTGREY
                tempNodeButton = Button(window=screen, color=color, x=nodeX, y=nodeY, width=140, height=100,
                                        text=nodeButton.text, shape='ellipse')
                refreshBackground(LIGHTGREY, WHITE)
                tempNodeButton.draw()
                pygame.display.update()
            pygame.time.wait(100)
            self.refreshTreeVisualizer(rootNode)

    def goBackToParent(self):
        if len(nodeStack) <= 1:
            return None
        nodeStack.pop()
        self.toggleMinimaxCurrentMode()

        rootY, parentY = rootNodeButton.y, parentNodeButton.y
        rootX = rootNodeButton.x
        while rootY not in range(int(parentY) - 3, int(parentY) + 3):
            if rootY > parentY:
                rootY -= 3
            color = BLUE
            tempRootButton = Button(window=screen, color=color, x=rootX, y=rootY, width=140, height=100,
                                    text=rootNodeButton.text, shape='ellipse')
            refreshBackground(LIGHTGREY, WHITE)
            tempRootButton.draw()
            pygame.display.update()
        pygame.time.wait(100)
        self.refreshTreeVisualizer(0)

    def drawMiniGameBoard(self, state=None):
        """
        Draws the game board on the interface with the latest values in the board list
        """
        if state is None:
            flippedGameBoard = convertToTwoDimensions(state=root)
            gameBoard = np.flip(m=flippedGameBoard, axis=0)
        else:
            flippedGameBoard = convertToTwoDimensions(state=state)
            gameBoard = np.flip(m=flippedGameBoard, axis=0)
        for i in range(ROW_COUNT):
            for j in range(COLUMN_COUNT):
                gameBoard[i][j] += 1

        MINISQUARESIZE = 30
        MINI_PIECE_RADIUS = MINISQUARESIZE / 2 - 2
        layout = pygame.draw.rect(surface=screen, color=LIGHTGREY,
                                  rect=(0, 0, MINISQUARESIZE * 7 + 40, MINISQUARESIZE * 6 + 40))
        #gradientRect(window=screen, left_colour=(40, 40, 40), right_colour=(25, 25, 25), target_rect=layout)
        pygame.draw.rect(screen, LIGHTGREY, (20, 20, MINISQUARESIZE * 7, MINISQUARESIZE * 6), 0)
        for c in range(COLUMN_COUNT):
            for r in range(ROW_COUNT):
                col = 20 + c * MINISQUARESIZE
                row = 20 + r * MINISQUARESIZE
                piece = gameBoard[r][c]
                pygame.draw.rect(
                    screen, CELL_BORDER_COLOR, (col, row, MINISQUARESIZE, MINISQUARESIZE))
                pygame.draw.circle(
                    screen, PIECE_COLORS[piece],
                    (int(col + MINISQUARESIZE / 2), int(row + MINISQUARESIZE / 2)), MINI_PIECE_RADIUS)
        pygame.display.update()

    def drawTreeVisualizerButtons(self):
        global backButton, backIcon, backIconAccent

        backIconAccent = pygame.image.load('back-icon.png').convert_alpha()
        backIcon = pygame.image.load('back-icon-accent.png').convert_alpha()

        backButton = Button(window=screen, color=WHITE, x=WIDTH - 70, y=20, width=52, height=52)
        self.reloadBackButton(backIcon)

    def drawTreeVisualizerLabels(self):
        labelFont = pygame.font.SysFont("Sans Serif", 55, False, True)
        modeLabel = labelFont.render(minimaxCurrentMode, True, BLACK)
        screen.blit(modeLabel,
                    (rootNodeButton.x + rootNodeButton.width + 20,
                     rootNodeButton.y + rootNodeButton.height / 2 - modeLabel.get_height() / 2))

    def toggleMinimaxCurrentMode(self):
        global minimaxCurrentMode
        if minimaxCurrentMode == "MAX":
            minimaxCurrentMode = "MIN"
        else:
            minimaxCurrentMode = "MAX"

    def reloadBackButton(self, icon):
        backButton.draw()
        screen.blit(icon, (backButton.x + 2, backButton.y + 2))

    def buttonResponseToMouseEvent(self, event):

        if event.type == pygame.MOUSEMOTION:
            if backButton.isOver(event.pos):
                pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
                self.reloadBackButton(backIconAccent)
            elif parentNodeButton.isOver(event.pos):
                if len(nodeStack) > 1:
                    pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
                    parentNodeButton.draw(WHITE, fontColor=WHITE)
                    pygame.display.update()
            elif rootNodeButton.isOver(event.pos):
                pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
                self.hoverOverNode(nodeButton=rootNodeButton, nodeState=root)
            elif child1Button.isOver(event.pos):
                if child1 is not None:
                    pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
                    self.hoverOverNode(nodeButton=child1Button, nodeState=child1)
            elif child2Button.isOver(event.pos):
                if child2 is not None:
                    pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
                    self.hoverOverNode(nodeButton=child2Button, nodeState=child2)
            elif child3Button.isOver(event.pos):
                if child3 is not None:
                    pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
                    self.hoverOverNode(nodeButton=child3Button, nodeState=child3)
            elif child4Button.isOver(event.pos):
                if child4 is not None:
                    pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
                    self.hoverOverNode(nodeButton=child4Button, nodeState=child4)
            elif child5Button.isOver(event.pos):
                if child5 is not None:
                    pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
                    self.hoverOverNode(nodeButton=child5Button, nodeState=child5)
            elif child6Button.isOver(event.pos):
                if child6 is not None:
                    pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
                    self.hoverOverNode(nodeButton=child6Button, nodeState=child6)
            elif child7Button.isOver(event.pos):
                if child7 is not None:
                    pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
                    self.hoverOverNode(nodeButton=child7Button, nodeState=child7)
            else:
                pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)
                self.reloadBackButton(backIcon)
                self.refreshTreeVisualizer(rootNode=0)
                pygame.display.update()

        if event.type == pygame.MOUSEBUTTONDOWN:
            if backButton.isOver(event.pos):
                gameWindow = GameWindow()
                gameWindow.switch()
            if parentNodeButton.isOver(event.pos):
                self.goBackToParent()
            elif child1Button.isOver(event.pos) and self.isNavigable(child1Button.text):
                self.navigateNode(node=child1, rootNode=root, nodeButton=child1Button)
            elif child2Button.isOver(event.pos) and self.isNavigable(child2Button.text):
                self.navigateNode(node=child2, rootNode=root, nodeButton=child2Button)
            elif child3Button.isOver(event.pos) and self.isNavigable(child3Button.text):
                self.navigateNode(node=child3, rootNode=root, nodeButton=child3Button)
            elif child4Button.isOver(event.pos) and self.isNavigable(child4Button.text):
                self.navigateNode(node=child4, rootNode=root, nodeButton=child4Button)
            elif child5Button.isOver(event.pos) and self.isNavigable(child5Button.text):
                self.navigateNode(node=child5, rootNode=root, nodeButton=child5Button)
            elif child6Button.isOver(event.pos) and self.isNavigable(child6Button.text):
                self.navigateNode(node=child6, rootNode=root, nodeButton=child6Button)
            elif child7Button.isOver(event.pos) and self.isNavigable(child7Button.text):
                self.navigateNode(node=child7, rootNode=root, nodeButton=child7Button)

            pygame.display.update()

        if event.type == pygame.MOUSEBUTTONUP:
            pass

    def hoverOverNode(self, nodeButton, nodeState=None):
        nodeButton.color = GREEN
        nodeButton.text = str(nodeState)
        if nodeState is not None and self.isPruned(nodeState):
            nodeButton.color = RED
        nodeButton.draw(fontSize=10)
        self.drawMiniGameBoard(nodeState)
        pygame.display.update()

    def isPruned(self, state):
        return state == 'PRUNED'                or int(state) & int('1000000000000000000000000000000000000000000000000000000000000000', 2) == 0

    def isNull(self, state):
        return state is None or state == ''

    def isNavigable(self, state):
        return not self.isNull(state) and not self.isPruned(state)


# ### Settings Class

# In[16]:


class SettingsWindow:
    def switch(self):
        self.setupSettingsMenu()
        self.show()

    def show(self):
        while True:
            pygame.display.update()

            event_list = pygame.event.get()
            for event in event_list:
                if event.type == pygame.QUIT:
                    sys.exit()

                self.buttonResponseToMouseEvent(event)

            global HEURISTIC_USED
            selectedOption = heuristicComboBox.update(event_list)
            heuristicComboBox.draw(screen)
            if selectedOption != HEURISTIC_USED:
                HEURISTIC_USED = selectedOption if selectedOption != -1 else HEURISTIC_USED
                self.refreshSettingsMenu()

    def setupSettingsMenu(self):
        """
        Initializes the all components in the frame
        """
        pygame.display.flip()
        pygame.display.set_caption('Saad & Morougue Connect 4 - Settings')
        self.setupSettingsMenuButtons()
        self.refreshSettingsMenu()

    def refreshSettingsMenu(self):
        """
        Refreshes the screen and all the components
        """
        pygame.display.flip()
        # Draw Background Image
        background_image = pygame.image.load("backgroud.png")  
        screen.blit(background_image, (0, 0))  # Blit the image onto the screen at the specified position
        self.drawSettingsMenuButtons()
        self.drawSettingsMenuLabels()

    def drawSettingsMenuButtons(self):
        self.reloadBackButton(backIcon)
        self.togglePruningCheckbox(toggle=False)
        modifyDepthButton.draw(BLACK)
        heuristicComboBox.draw(screen)

    def setupSettingsMenuButtons(self):
        global backButton, modifyDepthButton, pruningCheckbox, backIcon, backIconAccent, heuristicComboBox

        backIconAccent = pygame.image.load('back-icon.png').convert_alpha()
        backIcon = pygame.image.load('back-icon-accent.png').convert_alpha()

        backButton = Button(window=screen, color=LIGHTGREY, x=30, y=680, width=52, height=52)
        self.reloadBackButton(backIcon)

        pruningCheckbox = Button(
            screen, color=WHITE,
            x=830, y=400,
            width=30, height=30, text="",
            gradCore=usePruning, coreLeftColor=YELLOW, coreRightColor=BLACK,
            gradOutline=True, outLeftColor=RED, outRightColor=BLACK)
        self.togglePruningCheckbox(toggle=False)

        modifyDepthButton = Button(
            screen, color=RED,
            x=30, y=400,
            width=200, height=50, text="Choose Search Depth K")
        modifyDepthButton.draw(BLACK)

        heuristicComboBox = OptionBox(x=400, y=400,
                                      width=200, height=50, color=RED, highlight_color=YELLOW,
                                      selected=HEURISTIC_USED,
                                      font=pygame.font.SysFont("comicsans", 15),
                                      option_list=['Meduim', 'Hard'])
        heuristicComboBox.draw(screen)

    def reloadBackButton(self, icon):
        backButton.draw()
        screen.blit(icon, (backButton.x + 2, backButton.y + 2))

    def togglePruningCheckbox(self, toggle=True):
        global usePruning
        if toggle:
            usePruning = pruningCheckbox.isChecked = pruningCheckbox.gradCore = not usePruning

        if usePruning:
            pruningCheckbox.draw(RED, outlineThickness=4)
        else:
            pruningCheckbox.draw(RED, outlineThickness=2)

    def drawSettingsMenuLabels(self):
        global aiSettingsHR

        titleFont = pygame.font.SysFont("Sans Serif", 65, False, True)
        captionFont1_Arial = pygame.font.SysFont("Arial", 16)
        captionFont2_Arial = pygame.font.SysFont("Arial", 23)
        captionFont2_SansSerif = pygame.font.SysFont("Sans Serif", 23)

        mainLabel = titleFont.render("Game Settings", True, BLACK)
        pruningCaption = captionFont1_Arial.render("Use ALPHA-BETA Pruning", True, BLACK)
        depthCaption = captionFont2_Arial.render("k = " + str(BOARD.getDepth()), True, BLACK)
        heuristicCaption = captionFont2_Arial.render("Choose Difficulty (heuristic)", True, BLACK)
        backLabel = captionFont2_SansSerif.render("BACK", True, YELLOW)

        screen.blit(backLabel, (backButton.x + 5, backButton.y + backButton.height + 8))

        screen.blit(mainLabel, (WIDTH / 2 - mainLabel.get_width() / 2, HEIGHT / 8))

        screen.blit(pruningCaption,
                    (pruningCheckbox.x + pruningCheckbox.width + 10,
                     pruningCheckbox.y + pruningCaption.get_height() / 3))


        screen.blit(depthCaption,
                    (modifyDepthButton.x + modifyDepthButton.width + 10,
                     modifyDepthButton.y + depthCaption.get_height() / 3))

        screen.blit(heuristicCaption,
                    (heuristicComboBox.rect.x + heuristicComboBox.rect.width + 10,
                     heuristicComboBox.rect.y + depthCaption.get_height() / 3))

    def buttonResponseToMouseEvent(self, event):
        """
        Handles button behaviour in response to mouse events influencing them
        """
        if event.type == pygame.MOUSEMOTION:
            if modifyDepthButton.isOver(event.pos):
                pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
                alterButtonAppearance(modifyDepthButton, YELLOW, BLACK, 4)
            elif pruningCheckbox.isOver(event.pos):
                pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
            elif backButton.isOver(event.pos):
                pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
                self.reloadBackButton(backIconAccent)
            else:
                pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)
                alterButtonAppearance(modifyDepthButton, RED, BLACK)
                self.reloadBackButton(backIcon)

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if modifyDepthButton.isOver(event.pos):
                alterButtonAppearance(modifyDepthButton, YELLOW, BLACK)
            elif pruningCheckbox.isOver(event.pos):
                self.togglePruningCheckbox()
            elif backButton.isOver(event.pos):
                if gameInSession:
                    gameWindow = GameWindow()
                    gameWindow.switch()
                else:
                    mainMenu.switch()

        elif event.type == pygame.MOUSEBUTTONUP:
            if modifyDepthButton.isOver(event.pos):
                alterButtonAppearance(modifyDepthButton, RED, BLACK)
                self.takeNewDepth()

    def takeNewDepth(self):
        """
        Invoked at pressing modify depth button. Displays a simple dialog that takes input depth from user
        """
        temp = simpledialog.askinteger('Enter depth', 'Enter depth k')
        if temp is not None and temp > 0:
            BOARD.setDepth(temp)
        self.refreshSettingsMenu()


# ### Button Class

# In[17]:


class Button:
    def __init__(self, window, color, x, y, width, height, text='', isChecked=False, gradCore=False, coreLeftColor=None,
                 coreRightColor=None, gradOutline=False, outLeftColor=None, outRightColor=None, shape='rect'):
        self.color = color
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = text
        self.screen = window
        self.isChecked = isChecked
        self.gradCore = gradCore
        self.coreLeftColor = coreLeftColor
        self.coreRightColor = coreRightColor
        self.gradOutline = gradOutline
        self.outLeftColor = outLeftColor
        self.outRightColor = outRightColor
        self.shape = shape

    def draw(self, outline=None, outlineThickness=2, font='comicsans', fontSize=15, fontColor=BLACK):
        """
        Draws the button on screen
        """
        if self.shape.lower() == 'rect':
            if outline:
                rectOutline = pygame.draw.rect(self.screen, outline, (self.x, self.y,
                                                                      self.width, self.height), 0)
                if self.gradOutline:
                    gradientRect(self.screen, self.outLeftColor, self.outRightColor, rectOutline)
            button = pygame.draw.rect(self.screen, self.color, (self.x + outlineThickness, self.y + outlineThickness,
                                                                self.width - 2 * outlineThickness,
                                                                self.height - 2 * outlineThickness), 0)
            if self.gradCore:
                gradientRect(self.screen, self.coreLeftColor, self.coreRightColor, button, self.text, font, fontSize)

            if self.text != '':
                font = pygame.font.SysFont(font, fontSize)
                text = font.render(self.text, True, fontColor)
                self.screen.blit(text, (
                    self.x + (self.width / 2 - text.get_width() / 2),
                    self.y + (self.height / 2 - text.get_height() / 2)))
        elif self.shape.lower() == 'ellipse':
            if outline:
                rectOutline = pygame.draw.ellipse(self.screen, outline, (self.x, self.y,
                                                                         self.width, self.height), 0)
            button = pygame.draw.ellipse(self.screen, self.color, (self.x + outlineThickness, self.y + outlineThickness,
                                                                   self.width - 2 * outlineThickness,
                                                                   self.height - 2 * outlineThickness), 0)
            if self.text != '':
                font = pygame.font.SysFont(font, fontSize)
                text = font.render(self.text, True, fontColor)
                self.screen.blit(text, (
                    self.x + (self.width / 2 - text.get_width() / 2),
                    self.y + (self.height / 2 - text.get_height() / 2)))
        else:
            button = pygame.draw.circle(self.screen, self.color, (self.x + outlineThickness, self.y + outlineThickness,
                                                                  self.width - 2 * outlineThickness,
                                                                  self.height - 2 * outlineThickness), 0)
        return self, button

    def isOver(self, pos):
        # Pos is the mouse position or a tuple of (x,y) coordinates
        if self.x < pos[0] < self.x + self.width:
            if self.y < pos[1] < self.y + self.height:
                return True

        return False


# ### Option Box Class

# In[18]:


class OptionBox:

    def __init__(self, x, y, width, height, color, highlight_color, option_list, font, selected=0):
        self.color = color
        self.highlight_color = highlight_color
        self.rect = pygame.Rect(x, y, width, height)
        self.font = font
        self.option_list = option_list
        self.selected = selected
        self.draw_menu = False
        self.menu_active = False
        self.active_option = -1

    def draw(self, surf):
        pygame.draw.rect(surf, self.highlight_color if self.menu_active else self.color, self.rect)
        pygame.draw.rect(surf, (0, 0, 0), self.rect, 2)
        msg = self.font.render(self.option_list[self.selected], 1, (0, 0, 0))
        surf.blit(msg, msg.get_rect(center=self.rect.center))

        if self.draw_menu:
            for i, text in enumerate(self.option_list):
                rect = self.rect.copy()
                rect.y += (i + 1) * self.rect.height
                pygame.draw.rect(surf, self.highlight_color if i == self.active_option else self.color, rect)
                msg = self.font.render(text, 1, (0, 0, 0))
                surf.blit(msg, msg.get_rect(center=rect.center))
            outer_rect = (
                self.rect.x, self.rect.y + self.rect.height, self.rect.width, self.rect.height * len(self.option_list))
            pygame.draw.rect(surf, (0, 0, 0), outer_rect, 2)

    def update(self, event_list):
        mpos = pygame.mouse.get_pos()
        self.menu_active = self.rect.collidepoint(mpos)

        self.active_option = -1
        if self.draw_menu:
            for i in range(len(self.option_list)):
                rect = self.rect.copy()
                rect.y += (i + 1) * self.rect.height
                if rect.collidepoint(mpos):
                    self.active_option = i
                    break

        if not self.menu_active and self.active_option == -1:
            self.draw_menu = False

        for event in event_list:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.menu_active:
                    self.draw_menu = not self.draw_menu
                elif self.draw_menu and self.active_option >= 0:
                    self.selected = self.active_option
                    self.draw_menu = False
                    return self.active_option
        return -1


# In[ ]:


def gradientRect(window, left_colour, right_colour, target_rect, text=None, font='comicsans', fontSize=15):
    """
    Draw a horizontal-gradient filled rectangle covering <target_rect>
    """
    colour_rect = pygame.Surface((2, 2))  # 2x2 bitmap
    pygame.draw.line(colour_rect, left_colour, (0, 0), (0, 1))
    pygame.draw.line(colour_rect, right_colour, (1, 0), (1, 1))
    colour_rect = pygame.transform.smoothscale(colour_rect, (target_rect.width, target_rect.height))
    window.blit(colour_rect, target_rect)

    if text:
        font = pygame.font.SysFont(font, fontSize)
        text = font.render(text, True, (0, 0, 0))
        window.blit(text, (
            target_rect.x + (target_rect.width / 2 - text.get_width() / 2),
            target_rect.y + (target_rect.height / 2 - text.get_height() / 2)))


def alterButtonAppearance(button, color, outlineColor, outlineThickness=2,
                          hasGradBackground=False, gradLeftColor=None, gradRightColor=None, fontSize=15):
    """
    Alter button appearance with given colors
    """
    button.color = color
    thisButton, buttonRect = button.draw(outline=outlineColor, outlineThickness=outlineThickness)
    if hasGradBackground:
        gradientRect(screen, gradLeftColor, gradRightColor, buttonRect, thisButton.text, 'comicsans', fontSize)


def refreshBackground(leftColor=BLACK, rightColor=GREY):
    """
    Refreshes screen background
    """
    gradientRect(screen, leftColor, rightColor, pygame.draw.rect(screen, SCREEN_BACKGROUND, (0, 0, WIDTH, HEIGHT)))


def switchTurn():
    """
    Switch turns between player 1 and player 2
    """
    global TURN
    if TURN == 1:
        TURN = 2
    else:
        TURN = 1


def startGameSession():
    gameWindow = GameWindow()
    gameWindow.setupGameWindow()
    gameWindow.gameSession()


def setGameMode(mode):
    global GAME_MODE
    GAME_MODE = mode


if __name__ == '__main__':
    pygame.init()
    mainMenu = MainMenu()
    mainMenu.setupMainMenu()
    mainMenu.show()

