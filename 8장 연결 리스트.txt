# <<<<<<<<<<8장. 연결 리스트>>>>>>>>>>

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


