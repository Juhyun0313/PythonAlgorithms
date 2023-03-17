# <<<<<<<<<<6장. 문자열 조작>>>>>>>>>>

import collections

str1 = "A man, a plan, a canal: Panama"
class function :
    def __init__(self):
        print("클래스 선언!")
    def isPalindrome1(self, s: str) -> bool :
        strs = []
        for char in s :                   # str1의 문자를 하나씩 가져온다.
            if char.isalnum():            # char가 alphabet인지, number인지 확인
                strs.append(char.lower()) # 맞으면 빈 리스트에 하나씩 소문자로 변환하여 추가
            else:
                pass
        
        while len(strs) > 1 :             # pop 매서드로 가장 앞뒤의 문자를 출력하고 제거  
            if strs.pop(0) != strs.pop(): # 출력 당시 두 문자가 다르면 False 출력
                return False
            
        return True

    def isPalindrome2(self, s: str) -> bool :
        strs: Deque = collections.deque()
        
        for char in s:
            if char.isalnum():
                strs.append(char.lower())
        
        while len(strs) > 1:
            if strs.popleft() != strs.pop():
                return False
        
        return True
    
    def isPalindrome3(self, s: str) -> bool :
        s = s.lower()
        
        # 정규식 표현을 통한 문자열 필터링
        s = re.sub('[^a-z0-9]', '', s)
        
        return s == s[::-1]    # s[::-1] -> 문자열 뒤집기
        
        
        
import re
        
a = function()
a.isPalindrome1("A man, a plan, a canal: Panama")
a.isPalindrome2("A man, a plan, a canal: Panama")
a.isPalindrome3("race a car")
s = "race a car"
s = s.lower()
s = re.sub('[^a-z0-9]', '', s)
s == s[::-1]

def isPalindrome3(s: str) -> bool :
    s = s.lower()
    
    # 정규식 표현을 통한 문자열 필터링
    s = re.sub('[^a-z0-9]', '', s)
      
    return s == s[::-1]    # s[::-1] -> 문자열 뒤집기




class func :
    def ReverseString(self, li : list) -> None:
        for i in range(0,round(len(li)/2)):
            a = li[i]
            li[i] = li[-i-1]
            li[-i-1] = a
    
    def reverseString(self, s: list[str]) -> None:
        left, right = 0, len(s) -1
        while left < right:
            s[left], s[right] = s[right], s[left]
            left += 1
            right -= 1
            
for i in range(0,round(len(li)/2)):
    a = li[i]
    li[i] = li[-i-1]
    li[-i-1] = a  

li = ['h', 'e', 'l', 'l', 'o']
li = ['H', 'a', 'n', 'n', 'a', 'h']
li.reverse()
li
a = func()
a.reverseString(li)

logs = ["digl 8 1 5 1", "let1 art can", "dig2 3 6", "let2 own kit dig", "let3 art zero"]

logs[0].isdigit()

class funcs :
    def SortLogs(self, s : list[str]) -> list[str]:
        digits = []
        letters = []
        for logs in s:
            if logs.split()[1].isdigit():
                digits.append(logs)
            else:
                letters.append(logs)
        
        letters.sort(key = lambda x : (x.split()[1], x.split()[0]))
        
        return letters + digits

a = funcs()    
a.SortLogs(logs)    
    
paragraph = "Bob hit a ball, the hit BALL flew far after it was hit."
banned = ["hit"]  

import re
import pandas as pd


list(pd.DataFrame(re.sub('[^a-z0-9 ]', '', paragraph.lower()).split()).value_counts().drop(banned, level = 0).idxmax())



def MostCommonWord(self, s :str, ban : str) -> str:
    string = s.lower()
    trans = re.sub('[^a-z0-9 ]', '', string).split()  
    trans_df = pd.DataFrame(trans)
    
    return list(trans_df.value_counts().drop(banned, level = 0).idxmax())
    
    
MostCommonWord(paragraph, banned)    
    

group = ['eat', 'tea', 'tan', 'ate', 'nat', 'bat']
group.sort()
group
a = ['e', 'a', 't']
b = ['t', 'e', 'a']

sorted(a) == sorted(b)
a = 'eat'
a[0]
len(a)
wordsplit['a'] = []
wordsplit['a'].append('b')
wordsplit['b'].append('a')
sorted('eat')

def Anagrams(self, group : list(str)) -> list:
    wordsplit = collections.defaultdict(list)
    for words in group:
        # 정렬한 결과가 같은 값들끼리 묶어서 저장
        wordsplit[''.join(sorted(words))].append(words)
        
    return list(wordsplit.values())

'abc' == 'abc'[::-1]
s = 'babad'
a[2:3:] == a[2::-1]
a[2:1:-1]

def LongPalindrom(s : str) -> str:
    # 주어진 문자열에 대해서 1~len(s)까지 문자를 추출하여서
    # 팰린드롬인지 확인한다.
    for wordlen in range(1, len(s)):
        for i in range(0, len(s) +1 -wordlen):
            string = s[i:i+wordlen]
            if string == string[::-1]:
                result = string
    return result

class func :
    def LongestPalindrom(self, s : str) -> str:
        # 해당 사항이 없을 때 빠르게 리턴
        if len(s) < 2 or s == s[::-1]:
            return s
        
        result = ''
        # 슬라이딩 윈도우 우측으로 이동
        for i in range(len(s) - 1):
            result = max(result,
                         expand(i, i+1),
                         expand(i, i+2),
                         key = len)
            
        return result
LongPalindrom('abbddbb')
a = func()
a.LongestPalindrom('abbddbb')