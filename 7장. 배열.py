# <<<<<<<<<<7장. 배열>>>>>>>>>>


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