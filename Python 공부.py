# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 01:13:22 2022

@author: ahstm
"""
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



nums = [2, 7, 11, 15]
target = 9

def TargetIndex(nums : list[int], target : int) -> list[int]:
    for i in range(0, len(nums)):
        for j in range(1, len(nums) - i):
            if nums[i] + nums[i + j] == target:
                return [i, i + j]
    
TargetIndex(nums, target)

def twoSum(nums : list[int], target : int) -> list[int]:
    for i in range(0, len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
            
            
def twoSum(nums : list[int], target : int) -> list[int]:
    for i in range(len(nums)):
        if target - nums[i] in nums:
            return [i, nums.index(target - nums[i])]
        
for i, n in enumerate(nums):
    print(i, n)


def twoSum(nums : list[int], target : int) -> list[int]:
    nums_map = {}
    # 키와 값을 바꿔서 딕셔너리로 저장
    for i, num in enumerate(nums):
        nums_map[num] = i
    
    # 타켓에서 첫 번째 수를 뺀 결과를 키로 조회
    for i, num in enumerate(nums):
        if target - num in nums_map and i != nums_map[target - num]:
            return [i, nums_map[target-num]]
            

def twoSum(nums : list[int], target : int) -> list[int]:
    nums_map = {}
    # 하나의 for 문으로 통합
    for i, num in enumerate(nums):
        if target - num in nums_map:
            return [nums_map[target - num], i]
        nums_map[num] = i


a = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]
max(a)
i = 7
a[i] == 3
nums_map
a[9]    

import time

# 주어진 배열을 딕셔너리로 저장하여 배열에서 가장 큰 숫자의 인덱스를 구한다.
# 그렇게 구한 인덱스를 기준으로 투 포인트를 이용해 해당 인덱스까지 따로 계산하여 더해준다.

def trap(height : list[int]) -> int:
    # 최고 value의 인덱스 위치 찾기
    # 주어진 배열에서 max 값이 아니면 계속 반복
    i = 0
    while height[i] != max(height):
        i += 1
    
    # 투 포인터를 활용해서 최고항까지 연산
    left, right = 0, len(height) - 1
    
    # 왼쪽 포인터 계산
    leftValue = 0
    leftCurrentMax = 0
    while left <= i :
        if leftCurrentMax < height[left]:
            leftCurrentMax = height[left]
        else:
            leftValue += leftCurrentMax - height[left]
        left += 1
    
    # 오른쪽 포인터 계산
    rightValue = 0
    rightCurrentMax = 0
    while right >= i :
        if rightCurrentMax < height[right]:
            rightCurrentMax = height[right]
        else:
            rightValue += rightCurrentMax - height[right]
        right -= 1
    
    return leftValue + rightValue
        
        
def trap( height: list[int]) -> int:
    if not height:
        return 0
    
    volume = 0
    left, right = 0, len(height) -1
    left_max, right_max = height[left], height[right]
    
    while left < right:
        left_max, right_max = max(height[left], left_max),\
                              max(height[right], right_max)
        # 더 높은 쪽을 향해 투 포인트 이동
        if left_max <= right_max:
            volume += left_max - height[left]
            left += 1
        else:
            volume += right_max - height[right]
            right -= 1
    return volume

start = time.time()
trap([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1])
end = time.time()
print(end - start)

[0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]
def trap(self, height : list[int]) -> int:
    stack = []
    volume = 0
    
    for i in range(len(height)):
        # 변곡점을 만나는 경우
        # 스택이 빈 배열이 아닌 경우
        while stack and height[i] > height[stack[-1]]:
            # 스택에서 꺼낸다.
            top = stack.pop()
            
            if not len(stack):
                break
            
            # 이전과의 차이만큼 물 높이 처리
            distance = i - stack[-1] - 1
            waters = min(height[i], height[stack[-1]]) - height[top]
            
            volume += distance * waters
            
        stack.append(i)
    return volume

nums = [-1, 0, 1, 2, -1, -4]
def threeSum(nums: list[int]) -> list[list[int]]:
    # 배열에서 하나씩 추출하고 *-1 값이 나머지 배열 조합에서 있는지 확인
    result = []
    while len(nums) > 3:
        num = nums.pop(0)
        # 
        for i in range(len(nums)):
            if (nums[i] + num)*-1 in nums[i+1:]:
                for n in range(nums[i+1:].count((nums[i] + num)*-1)):
                    index = nums[i+1:].index((nums[i] + num)*-1)
                    result.append([num, nums[i], nums[index + i + 1]])
    # 마지막 남은 배열의 합이 0인지 확인
    if sum(nums) == 0:
        result.append(nums)
    
    return result
                
threeSum(nums)
nums.index(1)

def threeSum(nums: list[int]) -> list[list[int]]:
    # 배열에서 하나씩 추출하고 *-1 값이 나머지 배열 조합에서 있는지 확인
    result = []
    while len(nums) > 3:
        num = nums.pop(0)
        for i in range(len(nums)):
            if (nums[i] + num)*-1 in nums[i+1:]:
                index = nums[i+1:].index((nums[i] + num)*-1)
                if sorted([num, nums[i], nums[index + i + 1]]) not in result:
                    result.append(sorted([num, nums[i], nums[index + i + 1]]))

    # 마지막 남은 배열의 합이 0인지 확인
    if sum(nums) == 0 and sorted(nums) not in result:
        result.append(sorted(nums))
    
    return result

nums = [-1, 0, 1, 2, -1, -4]
nums.sort()
threeSum(nums)
threeSum([0,0,0,0])

def threeSum(nums: list[int]) -> list[list[int]]:
    results = []
    nums.sort()
    
    # 브루트 포스 n^3 반복
    for i in range(len(nums) - 2):
        # 중복된 값 건너뛰기
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        for j in range(i + 1, len(nums) - 1):
            if j > i + 1 and nums[j] == nums[j - 1]:
                continue
            for k in range(j + 1, len(nums)):
                if k > j + 1 and nums[k] == nums[k - 1]:
                    continue
                if nums[i] + nums[j] + nums[k] == 0:
                    results.append([nums[i], nums[j], nums[k]])
                    
    return results

def threeSum(nums: list[int]) -> list[list[int]]:
    results = []
    nums.sort()
    
    for i in range(len(nums) - 2):
        # 중복된 값 건너뛰기
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        # 간격을 좁혀가며 합 sum 계산
        left, right = i + 1, len(nums) - 1
        while left < right:
            sum = nums[i] + nums[left] + nums[right]
            if sum < 0:
                left += 1
            elif sum > 0:
                right -= 1
            else :
                # sum = 0인 경우이므로 정답 및 스킵 처리
                results.append([nums[i], nums[left], nums[right]])
                
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                left += 1
                right -= 1
                
    return results


def arrayPairSum(nums : list[int]) -> int:
    # 배열을 먼저 정렬
    nums.sort()
    
    result = 0
    right = len(nums) - 1
    while right >= 1 :
        result += min(nums[right], nums[right - 1])
        right -= 2
    
    return result

nums = [1,4,3,2]
arrayPairSum(nums)

def arrayPairSum(nums : list[int]) -> int:
    sum = 0
    pair = []
    nums.sort()
    
    for n in nums:
        # 앞에서부터 오름차순으로 페어를 만들어서 합 계산
        pair.append(n)
        if len(pair) == 2:
            sum += min(pair)
            pair = []
            
    return sum


def arrayPairSum(nums : list[int]) -> int:
    sum = 0
    nums.sort()
    
    for i, n in enumerate(nums):
        # 짝수 번째 값의 합 계산
        if i % 2 == 0:
            sum += n
    return sum

def arrayPairSum(nums : list[int]) -> int:
    return sum(sorted(nums)[::2]) 


def arrayMul(nums : list[int]) -> int:
    results = []
    for i in range(len(nums)):
        multiValue = 1
        for n in range(len(nums)):
            if i == n:
                continue
            multiValue *= nums[n]
        results.append(multiValue)
    return results

arrayMul(nums)
def arrayMul(nums : list[int]) -> int:
    out = []
    p = 1
    # 왼쪽 곱셈
    for i in range(len(nums)):
        out.append(p)
        p = p * nums[i]
    p = 1  # 재활용을 통해 공간복잡도를 O(n)에서 O(1)로 줄임
    # 왼쪽 곱셈 결과에 오른쪽 값을 차례대로 곱셈
    for i in range(len(nums) - 1, 0 - 1, -1):
        out[i] = out[i] * p
        p = p*nums[i]
    return out

def maxProfit(prices : list[int]) -> int:
    # 가장 작은 수와 가장 큰 수의 인덱스를 확인하여
    # 작은 수의 인덱스가 큰 수의 인덱스보다 작으면 바로 출력
    if prices.index(min(prices)) < prices.index(max(prices)):
        return max(prices) - min(prices)
    
    # 자신보다 오른쪽에 있는 숫자 중 가장 큰 수와의 차이
    result = 0
    for i in range(len(prices) - 1):
        if prices[i] < max(prices[i+1:]) and result < max(prices[i+1:]) - prices[i]:
            result = max(prices[i+1:]) - prices[i]
            
    return result

prices = [7,1,5,3,6,4]
maxProfit(prices)
prices[2:][::-1]
def maxProfit(prices : list[int]) -> int:
    
        result = 0
        curMaxIndex = -1
        preMaxIndex = 0
        
        if prices.index(max(prices)) > prices.index(min(prices)):
            return max(prices) - min(prices)
       
        while curMaxIndex < len(prices) - 1:
            curMax = max(prices[curMaxIndex+1:])
            curMaxIndex += len(prices[curMaxIndex+1:]) - prices[curMaxIndex+1:][::-1].index(curMax)
            
            curMin = min(prices[preMaxIndex:curMaxIndex+1])
            
            preMaxIndex = curMaxIndex
            
            if result < curMax - curMin:
                result = curMax - curMin
        
        return result
    
prices = [7,1,5,3,7,1]
len(prices) - prices[::-1].index(max(prices)) - 1



def maxProfit(prices : list[int]) -> int:
    max_price = 0
    
    for i, price in enumerate(prices):
         for j in range(i, len(prices)):
             max_price = max(prices[j] - price, max_price)
    
    return max_price

import sys
sys.maxsize
-sys.maxsize


def maxProfit(prices : list[int]) -> int:
    profit = 0
    min_price = sys.maxsize
    
    # 최솟값과 최댓값을 게속 갱신
    for price in prices:
        min_price = min(min_price, price)
        profit = max(profit, price - min_price)
    
    return profit

maxProfit

class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
head2 = ListNode(1)
head2.next = ListNode(3)
head2.next.next = ListNode(4)
head.next.next.next = ListNode(4)
def printNodes(node:ListNode):
    crnt_node = node
    while crnt_node is not None:
        print(crnt_node.val , end= ' ')
        crnt_node = crnt_node.next
        
printNodes(head1)
printNodes(head2)


def Palindrome(head : ListNode) -> bool:
    q: list = []
    
    # head에 아무것도 없을 때
    if not head:
        return False
    
    # head의 첫번째 노드를 시작으로 다음 노드를 배열로 저장
    node = head
    while node is not None:
        q.append(node.val)
        node = node.next
    
    # 팰린드롬 판별
    while len(q) > 1:
        if q.pop(0) != q.pop():
            return False
    return True

Palindrome(head)

def Palindrome(head : ListNode) -> bool:
    q = collections.deque()
    
    # head에 아무것도 없을 때
    if not head:
        return False
    
    # head의 첫번째 노드를 시작으로 다음 노드를 배열로 저장
    node = head
    while node is not None:
        q.append(node.val)
        node = node.next
    
    # 팰린드롬 판별
    while len(q) > 1:
        if q.popleft() != q.pop():
            return False
    return True

def Palindrome(head : ListNode) -> bool:
    rev = None
    slow = fast = head
    
    # 러너를 이용해 역순 연결 리스트 구성
    while fast and fast.next :
        fast = fast.next.next
        rev, rev.next, slow = slow, rev, slow.next
    if fast:
        slow = slow.next
    
    # 팰린드롬 여부 확인
    while rev and rev.val == slow. val:
        slow, rev = slow.next, rev.next
    # 끝까지 비교가 되었다면 slow와 rev는 None 일 것이다.
    return not rev
printNodes(head)
printNodes(fast)
printNodes(slow)
printNodes(rev)


l1 = ListNode(1)
l1.next = ListNode(2)
l1.next.next = ListNode(4)

l2 = ListNode(1)
l2.next = ListNode(3)
l2.next.next = ListNode(4)
        
printNodes(head1)
printNodes(head2)


# # 빈 리스트에 각각의 연결리스트를 배열로 저장
# q: list = []
# while head1 and head2 is not None:
#     q.append(head1.val)
#     q.append(head2.val)
#     head1 = head1.next
#     head2 = head2.next

# # 저장된 배열을 정렬
# q.sort()

# # 배열을 다시 연결리스트로 저장
# sortedhead = ListNode(q[0])
# for nodes in range(1, len(q)):
#     sortedhead = sortedhead.next
#     sortedhead = ListNode(q[nodes])

class func:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        if (not l1) or (l2 and l1.val > l2.val):
            l1, l2 = l2, l1
        if l1:
            l1.next = self.mergeTwoLists(l1.next, l2)
        return l1


printNodes(l1.next.next)
printNodes(l2)

l1 = ListNode(1)
l1.next = ListNode(2)
l1.next.next = ListNode(3)
l1.next.next.next = ListNode(4)
l1.next.next.next.next = ListNode(5)

printNodes(l1.next)
l2 = ListNode(1)
l2.next = ListNode(2)

def reverseListNodes(l1 : ListNode) -> ListNode:
    rev = None
    slow = l1
    
    while slow:
        rev, rev.next, slow = slow, rev, slow.next
    
    return printNodes(rev)


def reverseList(head: ListNode) -> ListNode:
    def reverse(node: ListNode, prev: ListNode = None):
        if not node:
            return printNodes(prev)
        next, node.next = node.next, prev
        return reverse(next, node)
    
    return reverse(head)



def reverseList(head: ListNode) -> ListNode:
    node, prev = head, None
    
    while node:
        next, node.next = node.next, prev
        prev, node = node, next
    
    return prev


l1 = ListNode(2)
l1.next = ListNode(4)
l1.next.next = ListNode(3)

l2 = ListNode(5)
l2.next = ListNode(6)
l2.next.next = ListNode(4)

# 역순으로 정의된 연결리스트라는 것은 첫번째 값이 일의 자리라는 것
# 결과값을 뽑아내는 연결 리스트를 하나 더 만든다는 생각을 하는 것보다
# 하나의 연결리스트에 다른 연결리스트 값을 더해가면서 갱신한다는 느낌으로 가는게 더 나을 듯하다.
def sumList(l1: ListNode, l2: ListNode) -> ListNode:
    while (l1 or l2) is not None :   # 두 리스트 모두 None일 때까지 반복
        # if (l1 is None) and (l2 is not None):
        #     l1, l2 = l2, l1
        if l1.val + l2.val < 10:
            l1.val += l2.val
            l1, l2 = l1.next, l2.next
            sumList(l1, l2)
        else:
            l1.val += l2.val - 10
            l1.next.val += 1
            l1, l2 = l1.next, l2.next
            sumList(l1, l2)
            
    return printNodes(l1)


