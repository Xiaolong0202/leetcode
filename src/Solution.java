import java.util.*;

public class Solution {


    public static void main(String[] args) {
        String a =  "你好傻都没打算";
        char[] a1 = a.toCharArray();
        Arrays.sort(a1);
        System.out.println(a1);
    }

    Solution() {
    }

    public void reverseArrary(int[] arr, int s, int d) {
        for (int i = s; i < (s + d + 1) / 2; i++) {
            int temp = arr[i];
            arr[i] = arr[d - i];
            arr[d - i] = temp;


            HashMap<Integer, Object> hashMap = new HashMap<>();
            hashMap.keySet();
        }
    }

    public int singleNumber(int[] nums) {
        int len = nums.length;
        HashMap<Integer, Object> map = new HashMap<Integer, Object>();
        for (int i = 0; i < len; i++) {
            int a = nums[i];
            if (map.containsKey(a)) map.remove(a);
            else map.put(a, null);
        }
        for (int key : map.keySet()) {
            return key;
        }
        return 1;
    }

    public void moveZeroes(int[] nums) {
        if (nums == null || nums.length == 0)
            return;
        int index = 0;
        //一次遍历，把非零的都往前挪
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] != 0)
                nums[index++] = nums[i];
        }
        //后面的都是0,
        while (index < nums.length) {
            nums[index++] = 0;
        }
    }

    //len/2
    public void rotate(int[][] matrix) {
        int len = matrix.length;
        for (int i = 0; i < len / 2; i++) {
            for (int j = i; j < len - 1 - 2 * i + i; j++) {
                int temp = matrix[i][j];

                matrix[i][j] = matrix[len - 1 - j][i];
                matrix[len - 1 - j][i] = matrix[len - 1 - i][len - 1 - j];
                matrix[len - 1 - i][len - 1 - j] = matrix[j][len - 1 - i];
                matrix[j][len - 1 - i] = temp;

            }
        }
    }

    public boolean isAnagram(String s, String t) {
        if (s.length() != t.length()) return false;
        HashMap<Character, Integer> map = new HashMap<Character, Integer>();
        for (int i = 0; i < s.length(); i++) {
            char c1 = s.charAt(i);
            char c2 = t.charAt(i);
            map.put(c1, map.getOrDefault(c1, 0) + 1);
            map.put(c2, map.getOrDefault(c2, 0) - 1);
        }

        for (Integer i : map.values()) {
            if (i != 0) return false;
        }
        return true;
    }

    public int myAtoi(String s) {
        s = s.trim();
        boolean isfu = false;
        char[] num = s.toCharArray();
        if (num.length == 0) return 0;
        int i = 0;
        int res = 0;

        if (num[i] == '-' || num[i] == '+') {
            if (num[i] == '-') isfu = true;
            i++;
        }

        for (; i < num.length; i++) {
            char c = num[i];
            if (c >= '0' && c <= '9') {
                int temp = res;
                res = res * 10 + c - '0';
                if (res / 10 != temp) {
                    if (isfu) return Integer.MIN_VALUE;
                    else return Integer.MAX_VALUE;
                }
            } else {
                break;
            }
        }
        return isfu ? -1 * res : res;
    }


    public boolean isPalindrome(String s) {
        s = s.toLowerCase();
        char[] c = s.toCharArray();

        int i = 0;
        int j = c.length - 1;

        while (i < j) {
            if (!Character.isLetterOrDigit(c[i])) {
                i++;
                continue;
            }
            if (!Character.isLetterOrDigit(c[j])) {
                j--;
                continue;
            }
            if (c[i] != c[j]) return false;
            i++;
            j--;
        }
        return true;
    }

    public int strStr(String haystack, String needle) {
        if (haystack.length() < needle.length()) return -1;
        int[] next = build_next(needle);
        int i = 0;
        int j = 0;
        int result = -1;
        char[] t = haystack.toCharArray();
        char[] s = needle.toCharArray();
        while (i < t.length) {
            if (s[j] == t[i]) {
                i++;
                j++;
            } else if (j > 0) {
                j = next[j - 1];
            } else {
                i++;
            }


            if (j == s.length) {
                result = i - s.length;
                break;
            }
        }
        return result;
    }

    public int[] build_next(String needle) {
        char[] s = needle.toCharArray();
        int[] next = new int[s.length];
        int i = 1;
        int j = 0;
        while (i < s.length) {
            if (s[i] == s[j]) {
                j++;
                next[i++] = j;
            } else if (j == 0) {
                next[i++] = 0;
            } else {
                j = next[j - 1];
            }
        }
        return next;
    }

    public String countAndSay(int n) {
        if (n == 1) return "1";
        char[] pre = countAndSay(n - 1).toCharArray();
        StringBuilder sb = new StringBuilder();
        int i = 0;
        while (i < pre.length) {
            char c = pre[i];
            int count = 0;
            while (i < pre.length) {
                if (c == pre[i]) {
                    i++;
                    count++;
                    if (i == pre.length) {
                        sb.append(count);
                        sb.append(c);
                        break;
                    }
                } else {
                    sb.append(count);
                    sb.append(c);
                    break;
                }
            }
        }
        return sb.toString();
    }

    public String longestCommonPrefix(String[] strs) {
        String result = strs[0];
        for (int i = 1; i < strs.length; i++) {
            if (strs[i].length() == 0) return "";
            for (int j = strs[i].length(); j > 0; j--) {
                String t = strs[i].substring(0, j);
                if (result.indexOf(t) == 0) {
                    result = t;
                    break;
                }
                if (j == 1) return "";
            }
        }
        return result;
    }

    public class ListNode {
        int val;
        ListNode next;

        ListNode(int val) {
            this.val = val;
        }

        ListNode(int val, ListNode next) {
            this.val = val;
            this.next = next;
        }
    }

    public void deleteNode(ListNode node) {
        node.val = node.next.val;
        node.next = node.next.next;
    }

    public boolean isPalindrome(ListNode head) {
        if (head == null || head.next == null) return true;
        ListNode fast = head.next;
        ListNode slow = head;
        while (fast.next != null) {
            slow = slow.next;
            fast = fast.next;
            if (fast.next == null) break;
            fast = fast.next;
            if (fast.next == null) break;
        }
        slow = slow.next;
        ListNode newNode = reverseList(slow);
        while (newNode != null) {
            if (head.val != newNode.val) return false;
            newNode = newNode.next;
            head = head.next;
        }
        return true;
    }

    //反转数组
    public ListNode reverseList(ListNode head) {
        ListNode preNode = null;
        while (head != null) {
            ListNode tempNode = head.next;
            head.next = preNode;
            preNode = head;
            head = tempNode;
        }
        return preNode;
    }

    public boolean hasCycle(ListNode head) {
        HashSet<ListNode> set = new HashSet<>();
        while (head != null) {
            if (head.next == head) return true;
            ListNode temp = head;
            head = head.next;
            temp.next = temp;
        }
        return false;
    }



    public int maxDepth(TreeNode root) {
        if (root == null) return 0;
        return Math.max(maxDepth(root.left) + 1, maxDepth(root.right) + 1);
    }

    public boolean isValidBST(TreeNode root) {
        return isValidBST(root, Long.MIN_VALUE, Long.MAX_VALUE);
    }

    public boolean isValidBST(TreeNode root, long minval, long maxval) {
        if (root == null) return true;
        if (root.val >= maxval || root.val <= minval) return false;
        return isValidBST(root.right, root.val, maxval) && isValidBST(root.left, minval, root.val);
    }

    public boolean isSymmetric(TreeNode root) {
        if (root == null) return true;
        return isSymmetric(root.left, root.right);
    }

    public boolean isSymmetric(TreeNode left, TreeNode right) {
        if (left == null && right == null) return true;
        if (left == null || right == null || left.val != right.val) return false;
        return isSymmetric(left.left, right.right) && isSymmetric(left.right, right.left);
    }

    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> list = new ArrayList<>();
        if (root == null) return list;
        ArrayDeque<TreeNode> deque = new ArrayDeque<>();
        deque.offerLast(root);
        while (!deque.isEmpty()) {
            int size = deque.size();//当前层的结点数量
            List<Integer> tempList = new ArrayList<>();
            while (size-- > 0) {
                TreeNode temp = deque.pollFirst();
                tempList.add(temp.val);
                if (temp.left != null) {
                    deque.offerLast(temp.left);
                }
                if (temp.right != null) {
                    deque.offerLast(temp.right);
                }
            }
            list.add(tempList);
        }
        return list;
    }

    public TreeNode sortedArrayToBST(int[] nums) {
        return sortedArrayToBST(nums, 0, nums.length - 1);
    }

    public TreeNode sortedArrayToBST(int[] nums, int start, int end) {
        if (start > end) return null;
        int mid = start + end >> 1;
        TreeNode node = new TreeNode(nums[mid]);
        node.left = sortedArrayToBST(nums, start, mid - 1);
        node.right = sortedArrayToBST(nums, mid + 1, end);
        return node;
    }

    public int maxSubArray(int[] nums) {
        int n = nums.length;
        if (n == 1) return nums[0];
        int dp;
        dp = nums[0];
        int reslut = dp;
        for (int i = 1; i < nums.length; i++) {
            if (dp + nums[i] > nums[i]) {
                dp = dp + nums[i];
            } else {
                dp = nums[i];
            }
            reslut = Math.max(reslut, dp);
        }
        return reslut;
    }

    //大家解设1
    public int rob1(int[] nums) {
        if (nums.length == 1) return nums[0];

        int[] dp = new int[nums.length];
        dp[0] = nums[0];
        dp[1] = Math.max(dp[0], nums[1]);
        int res = Math.min(dp[0], dp[1]);
        for (int i = 2; i < nums.length; i++) {
            dp[i] = Math.max(dp[i - 1], dp[i - 2] + nums[i]);
            res = Math.min(res, dp[i]);
        }
        return res;
    }


    private int pricese[];
    private Random random;

    public Solution(int[] nums) {
        this.pricese = nums;
        random = new Random();
    }

    public int[] reset() {
        return pricese;
    }

    public int[] shuffle() {
        int[] res = pricese.clone();
        for (int i = 0; i < pricese.length; i++) {
            int a = random.nextInt(i + 1);
            int temp = pricese[a];
            pricese[a] = pricese[i];
            pricese[i] = temp;
        }
        return res;
    }

    class MinStack {
        int[] stk;
        int tt;

        public MinStack() {
            stk = new int[30000];
            tt = 0;
        }

        public void push(int val) {
            stk[tt++] = val;
        }

        public void pop() {
            if (tt != 0) tt--;
        }

        public int top() {
            return stk[tt - 1];
        }

        public int getMin() {
            if (tt == 0) return 0;
            int res = Integer.MAX_VALUE;
            for (int i = tt - 1; i >= 0; i--) {
                res = Math.min(res, stk[i]);
            }
            return res;
        }
    }

    //试除法判断质数
    public boolean isPrime(int n) {
        for (int i = 2; i < n / i; i++) {
            if (n % i == 0) return false;
        }
        return true;
    }

    //埃氏筛求素数个数
    public int countPrimes(int n) {
        boolean[] notPrime = new boolean[n + 1];
        int count = 0;
        for (long i = 2; i < n; i++) {
            if (!notPrime[(int) i]) {//是素数
                for (long j = i * i; j <= n; j += i) {
                    notPrime[(int) j] = true;
                }
                count++;
            }
        }
        return count;
    }

    //快速幂
    public long quick_power(int base, int power) {
        long result = 1;
        while (power > 0) {
            if ((power & 1) != 0) result *= base;
            base *= base;
            power >>= 1;
        }
        return result;
    }

    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        int m = obstacleGrid.length;
        int n = obstacleGrid[0].length;
        int[][] dp = new int[m + 1][n + 1];
        dp[0][1] = 1;
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
                if (obstacleGrid[i - 1][j - 1] == 1) dp[i][j] = 0;
            }
        }
        return dp[m][n];
    }

    public int minPathSum(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;

        int[][] dp = new int[m + 1][n + 1];

        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (i == 1) dp[0][j] = Integer.MAX_VALUE;
                if (j == 1) dp[i][0] = Integer.MAX_VALUE;
                dp[i][j] = grid[i - 1][j - 1] + Math.min(dp[i - 1][j], dp[i][j - 1]);
                if (i == 1 && j == 1) dp[1][1] = grid[0][0];
                System.out.print(dp[i][j] + " ");
            }
            System.out.println();
        }
        return dp[m][n];
    }

    public int[] intersection(int[] nums1, int[] nums2) {
        HashSet<Integer> set = new HashSet<>();
        HashSet<Integer> set1 = new HashSet<>();
        for (int i = 0; i < nums1.length; i++) {
            set.add(nums1[i]);
        }
        for (int i = 0; i < nums2.length; i++) {
            if (set.contains(nums2[i])) set1.add(nums2[i]);
        }

        return set1.stream().mapToInt(x -> x).toArray();
    }

    public boolean isHappy(int n) {
        if (n == 1) return true;
        int m = n;
        while (m != 1) {
            n = getNext(n);
            m = getNext(getNext(m));
            if (m == n) return false;
        }
        return true;
    }

    public int getNext(int n) {
        int next = 0;
        while (n > 0) {
            int a = n % 10;
            System.out.println(a);
            n /= 10;
            next += a * a;
        }
        return next;
    }

    public int fourSumCount(int[] nums1, int[] nums2, int[] nums3, int[] nums4) {
        HashMap<Integer, Integer> map = new HashMap<>();
        int n = nums1.length;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int sum = nums1[i] + nums2[j];
                System.out.println(sum);
                map.put(sum, map.getOrDefault(sum, -1) + 1);
            }
        }
        System.out.println(map);
        int count = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int sum = nums3[i] + nums4[j];
                System.out.println(-sum);
                count += map.getOrDefault(-sum, 0);
            }
        }
        return count;
    }

    public List<List<Integer>> threeSum(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> list = new ArrayList<>();
        for (int i = 0; i < nums.length; i++) {
            int a = nums[i];
            if (a > 0) break;
            if ((i > 0 && a == nums[i - 1])) continue;
            int l = i + 1;
            int r = nums.length - 1;
            while (l < r) {
                if (nums[l] + nums[r] > -a) {
                    r--;
                } else if (nums[l] + nums[r] < -a) {
                    l++;
                } else {
                    List<Integer> integerList = new ArrayList<>();
                    integerList.add(a);
                    integerList.add(nums[l]);
                    integerList.add(nums[r]);
                    list.add(integerList);
                    while (l < r && nums[l + 1] == nums[l]) l++;
                    while (l < r && nums[r - 1] == nums[r]) r--;
                    l++;
                    r--;
                }
            }
        }
        return list;
    }

    public boolean canConstruct(String ransomNote, String magazine) {
        HashMap<Character, Integer> map = new HashMap<>();
        char[] chars1 = ransomNote.toCharArray();
        char[] chars2 = magazine.toCharArray();
        for (char c : chars1) {
            map.put(c, map.getOrDefault(c, 0) + 1);
        }
        for (char c : chars2) {
            if (map.containsKey(c)) {
                map.put(c, map.get(c) - 1);
            }
        }
        for (Integer i : map.values()) {
            if (i > 0) return false;
        }
        return true;
    }

    public List<List<Integer>> fourSum(int[] nums, int target) {
        List<List<Integer>> listList = new ArrayList<>();
        if (nums.length < 4) return listList;
        Arrays.sort(nums);
        for (int i = 0; i < nums.length; i++) {
            long a = nums[i];
            // if (a>target)break;
            if (i > 0 && a == nums[i - 1]) continue;
            for (int j = i + 1; j < nums.length; j++) {
                long b = nums[j];
                // if (a+b>target)break;
                if ((j > i + 1 && b == nums[j - 1])) continue;
                int l = j + 1;
                int r = nums.length - 1;
                while (l < r) {
                    // System.out.println(i+" "+j+" "+l+" "+r);
                    if (a + b + nums[l] + nums[r] > target) {
                        r--;
                    } else if (a + b + nums[l] + nums[r] < target) {
                        l++;
                    } else {
                        List<Integer> list = new ArrayList<>();
                        list.add((int) a);
                        list.add((int) b);
                        list.add(nums[l]);
                        list.add(nums[r]);
                        listList.add(list);
                        while (l < r && nums[l] == nums[l + 1]) l++;
                        while (l < r && nums[r] == nums[r - 1]) r--;
                        l++;
                        r--;
                    }
                }
            }
        }
        return listList;
    }

    //喂饼干
    public int findContentChildren(int[] g, int[] s) {
        Arrays.sort(g);
        Arrays.sort(s);
        int i = 0;
        int j = 0;
        int count = 0;
        while (i < g.length && j < s.length) {
            if (g[i] <= s[j]) {
                count++;
                i++;
                j++;
            } else {
                j++;
            }
        }
        return count;
    }

    //摆动数组
    public int wiggleMaxLength(int[] nums) {
        if (nums.length == 1) return 1;
        if (nums.length == 2 && nums[1] != nums[0]) return 2;
        int a = nums[0];
        int j = 1;
        int pre = nums[0] - nums[1];
        while (pre == 0 && nums.length > j) {
            pre = nums[j] - nums[j - 1];
            j++;
        }
        System.out.println(j);
        int count = 1;
        while (nums.length > j) {
            int minus = nums[j] - a;
            if (minus * pre < 0) {
                count++;
            }
            a = nums[j++];
            if (minus != 0) pre = minus;
        }
        return count;
    }

    //快速幂
    //快速幂
    public double myPow(double x, int n) {
        long a = n;
        if (n < 0) {
            return 1 / quickPow(x, -a);
        }
        return quickPow(x, a);
    }

    //快速幂
    public double quickPow(double x, long n) {
        double result = 1;
        while (n != 0) {
            if ((n & 1) != 0) result *= x;
            x *= x;
            n >>= 1;
        }
        return result;
    }

    public boolean canJump(int[] nums) {
        if (nums.length == 1 || nums.length == 0) return true;
        int k = 0;
        for (int i = 0; i < nums.length; i++) {
            if (k<i)return false;
            k = Math.max(k,nums[i]+i);
        }
        return true;
    }

    public int jump(int[] nums) {
        if (nums.length==1)return 0;
        int count = 0;
        int nextMaxWay= 0;
        int currentMaxWay = nums[0];
        for (int i = 1; i < nums.length; i++) {
            nextMaxWay = Math.max(nextMaxWay,nums[i]+i);
            if (i==currentMaxWay){
                count++;
                currentMaxWay=nextMaxWay;
            }
        }
        return  count;
    }

    public int largestSumAfterKNegations(int[] nums, int k) {
        Arrays.sort(nums);
        int i;
        int sum = 0;
        for (i = 0; i < nums.length; i++) {
            if (nums[i]<0&&k>0){
                nums[i]=-nums[i];
                k--;
                }
            sum+=nums[i];
        }
            if (k!=0&&k%2!=0){
                Arrays.sort(nums);
                sum-=nums[0]*2;
            }
        return sum;
    }
    public int canCompleteCircuit(int[] gas, int[] cost) {
        int len = gas.length;
        int qidian = 0;
        int youliang = 0;
        int i=0;
        boolean flag =true;
        while (true){
            if (flag) youliang+=gas[i]-cost[i];
            //当前位置为负数，下一步不能走
            if (youliang<0){
                qidian=(qidian+len-1)%len;
                youliang+=gas[qidian]-cost[qidian];;
                flag=false;
                if (qidian==i)return -1;
            }else {
                flag=true;
                i=(i+1)%len;
                if (i==qidian)return qidian;
            }
        }
    }

    //分糖果
    public int candy(int[] ratings) {
        int len = ratings.length;
        int [] res = new int[len];//表示每个孩子分到的糖果数
        res[0] =  1;
        for (int i = 1; i < len; i++) {
            if (ratings[i] > ratings[i-1]){
                res[i] = res[i-1]+1;
            }else {
                res[i] = 1;
            }
        }
        res[len-1] = res[len-1];
        for (int i = len-2; i >= 0 ; i--) {
            if (ratings[i] > ratings[i+1])res[i] = Math.max(res[i],res[i+1]+1);
        }
        System.out.println(Arrays.toString(res));
        int sum = 0;
        for (int i = 0; i < len; i++) {
            sum+=res[i];
        }
        return sum;
    }

    public boolean lemonadeChange(int[] bills) {
        int five = 0, ten = 0;
        for (int bill : bills) {
            if (bill == 5) {
                five++;
            } else if (bill == 10) {
                if (five == 0) {
                    return false;
                }
                five--;
                ten++;
            } else {
                if (five > 0 && ten > 0) {
                    five--;
                    ten--;
                } else if (five >= 3) {
                    five -= 3;
                } else {
                    return false;
                }
            }
        }
        return true;
    }

    public int[][] reconstructQueue(int[][] people) {
        Arrays.sort(people, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                if (o1[0] != o2[0]){
                    return o2[0] - o1[0];
                }else {
                    return o1[1]-o2[1];
                }
            }
        });
        List<int[]> list = new ArrayList<>();
        for (int i = 0; i < people.length; i++) {
            list.add(people[i][1],people[i]);
        }
        return list.toArray(new int[people.length][2]);
    }


    public int removeElement(int[] nums, int val) {
        if (nums.length==1&&nums[0]!=val) return 1;
        if (nums.length==0)return 0;
        int i = 0;
        int j = nums.length - 1;
        int count = 0;
        while (i<j){
            if (nums[i] == val){
                while (nums[j]==val&&i<j){
                    j--;
                }
                if (nums[j]==val)break;
                int temp = nums[i];
                nums[i] = nums[j];
                nums[j] = temp;
            }
            i++;
        }
        if (j== nums.length-1&&nums[j]!=val)i++;
        return i;
    }

    public int[] sortedSquares(int[] nums) {
        int [] result = new int[nums.length];
        int i;
        for ( i = 0; i < nums.length; i++) {
            if (nums[i]>0)break;
        }
        int l = i-1;
        int r = i;
        int j = 0;
        while (l>=0&&r<=nums.length-1){
            if (Math.abs(nums[l])<=Math.abs(nums[r])){
                result[j++]=nums[l]*nums[l];
                l--;
            }else {
                result[j++]=nums[r]*nums[r];
                r++;
            }
        }

        while (l>=0){
            result[j++]=nums[l]*nums[l];
            l--;
        }
        while (r<=nums.length-1){
            result[j++]=nums[r]*nums[r];
            r++;
        }
        return result;
    }


    public String replaceSpace(String s) {
        return s.replaceAll(" ","%20");
    }

    public String reverseWords(String s) {
        s=s.trim();
        StringBuilder sb  = new StringBuilder();
        char[] chars = s.toCharArray();
        int len = chars.length;
        int i  = len -1 ;
        int j =  len;
        while (i>=0){
            if (chars[i]==' '){
                if ((i!=len-1)&&chars[i+1]==' '){
                }else {
                    sb.append(s.substring(i+1,j));
                    sb.append(' ');
                }
                j=i;
            }else if (i==0){
                sb.append(s.substring(i,j));
            }
            i--;
        }

        return sb.toString();
    }


        List<List<Integer>> combineList = new ArrayList<>();
        public List<List<Integer>> combine(int n, int k) {
            for (int i = 1; i <= n ; i++) {
                combine(1,i,n,k,new ArrayList<>());
            }
            return combineList;
        }

        public void combine(int floor,int addNum,int n,int k,List<Integer> list){
            list.add(addNum);
            if (floor==k){
                List<Integer> nList = new ArrayList<Integer>(list);
                combineList.add(nList);
                return;
            }
            for (int i = addNum+1; i <= n; i++) {
                combine(floor+1,i,n,k,list);
                list.remove(list.size()-1);
            }
        }

    public int findMinArrowShots(int[][] points) {
        Arrays.sort(points, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return Integer.compare(o1[0],o2[0]);
            }
        });

        int right = points[0][1];
        int count = 1;

        for (int i = 1; i < points.length; i++) {
            if (points[i][0] <= right){
                right = Math.min(points[i][1],right);
            }else {
                right = points[i][1];
                count++;
            }
        }
        return count;
    }

    public int eraseOverlapIntervals(int[][] intervals) {
        Arrays.sort(intervals, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                if (o1[0]==o2[0])return Integer.compare(o1[1],o2[1]);
                else return Integer.compare(o1[0],o2[0]);
            }
        });

        int right = intervals[0][1];
        int count = 0;
        for (int i = 1; i < intervals.length; i++) {
            if (intervals[i][0]>=right){
                // System.out.println(right+" "+intervals[i][1]);
                right=intervals[i][1];
            }else {
                count++;
                right = Math.min(right,intervals[i][1]);
            }
        }
        return count;
    }

    public List<Integer> partitionLabels(String s) {
        char[] chars = s.toCharArray();
        List<Integer> list = new ArrayList<>();
        HashMap<Character,Integer> map = new HashMap<>();
        for (int i = 0; i < chars.length; i++) {
            map.put(chars[i],i);
        }
        int preR = -1;
        int right = map.get(chars[0]);
        for (int i = 1; i < chars.length; i++) {
            int temp = map.get(chars[i]);
            if (i>right){
                list.add(right-preR);
                preR = right;
            }
            right = Math.max(right,temp);
        }
        list.add(right-preR);
        return list;
    }

    public int[][] merge(int[][] intervals) {
            Arrays.sort(intervals, new Comparator<int[]>() {
                @Override
                public int compare(int[] o1, int[] o2) {
                    if (o1[0]!=o2[0]) return Integer.compare(o1[0],o2[0]);
                    else return Integer.compare(o1[1],o2[1]);
                }
            });
            List<int[]> list = new ArrayList<>();
            int left = intervals[0][0];
            int right = intervals[0][1];

        for (int i = 1; i < intervals.length; i++) {
            if (intervals[i][0]<=right){
                right = Math.max(intervals[i][1],right);
            }else {
                int[] a = new int[]{left,right};
                list.add(a);
                left = intervals[i][0];
                right = intervals[i][1];
            }
        }
        list.add(new int[]{left,right});
        int[][] ints = list.toArray(new int[list.size()][2]);
        return ints;
    }

    public int monotoneIncreasingDigits(int n) {
        String string = Integer.toString(n);
        char[] num = string.toCharArray();
        for (int i = 0; i < num.length-1; i++) {
            if (num[i]>num[i+1]){
                while (i>0&&num[i]==num[i-1]){
                    i--;
                }
                num[i++]++;
            while (i<num.length){
                num[i++]='9';
            }
            StringBuilder stringBuilder = new StringBuilder();
                for (char c : num) {
                    stringBuilder.append(c);
                }
            return Integer.parseInt(stringBuilder.toString());
            }
        }
        return n;
    }

    List<List<Integer>> combinationListSum3 = new ArrayList<>();
    public List<List<Integer>> combinationSum3(int k, int n) {
        combinationSum3(k,n,0,0,0,new ArrayList<>());
        return combinationListSum3;
    }

    public void combinationSum3(int k, int n,int floor,int currentNum,int sum,List<Integer> list) {
        if (sum>n)return;
        else if (sum==n&&k<floor)return;
        else if (k==floor){
            if (sum==n) combinationListSum3.add(new ArrayList<>(list));
            return;
        }else {
                int maxI = n>9?9:n-1;
                for (int i = currentNum+1; i <= maxI; i++) {
                    list.add(i);
                    combinationSum3(k,n,floor+1,i,i+sum,list);
                    list.remove(list.size()-1);
                }
            }
    }

    List<List<Integer>> subsetDupList = new ArrayList<>();
    public List<List<Integer>> subsetsWithDup(int[] nums) {
            Arrays.sort(nums);
            boolean [][] subsetArr = new boolean[11][21];
            List<Integer> list = new ArrayList<>();
            subsetDupList.add(list);
            subsetsWithDup(nums,list,-1,0,subsetArr);
            return subsetDupList;
    }

    public void subsetsWithDup(int[] nums,List<Integer> list,int i,int floor,boolean[][] subsetArr) {
        if (floor==nums.length)return;
        for (int j = i+1; j < nums.length; j++) {
            if (!subsetArr[floor][nums[j]+10]){
                list.add(nums[j]);
                subsetDupList.add(new ArrayList<>(list));
                subsetArr[floor][nums[j]+10]=true;
                subsetsWithDup(nums,list,j,floor+1,subsetArr);
                list.remove(list.size()-1);
//                subsetArr[floor][nums[j]]=false;
            }
        }
        Arrays.fill(subsetArr[floor],false);
    }

    List<List<Integer>> combinationList = new ArrayList<>();
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        Arrays.sort(candidates);
        combinationSum(candidates,target,-1,0,0,new ArrayList<>());
        return combinationList;
    }
    public void combinationSum(int[] candidates, int target,int i,int floor,int currentSum,List<Integer> list) {
        if (currentSum>target)return;
        if (currentSum==target){
            combinationList.add(new ArrayList<>(list));
            return;
        }
        for (int j = i+1; j < candidates.length; j++) {
            int countPlusTime = 0;
            int a = candidates[j]+currentSum;
            while (a<=target){
                list.add(candidates[j]);
                combinationSum(candidates,target,j,floor+1,a,list);
                countPlusTime++;
                a+=candidates[j];
            }
            for (int k = 0; k < countPlusTime; k++) {
                list.remove(list.size()-1);
            }
        }
    }

    public int[][] generateMatrix(int n) {
        int[][] res = new int[n][n];
        generateMatrix(0,n,res,1);
        return res;
    }

    public void generateMatrix(int m,int n,int[][] res,int currentNum) {
        for (int i = m; i < n-m; i++) {
            res[m][i]=currentNum++;
        }

        for (int i = m+1; i < n-m; i++) {
            res[i][n-m-1]=currentNum++;
        }

        for (int i = n-m-2 ; i >= m; i--) {
            res[n-m-1][i]=currentNum++;
        }

        for (int i = n-m-2; i > m; i--) {
            res[i][m] = currentNum++;
        }
        if (currentNum==n*n+1)return;
        generateMatrix(m+1,n,res,currentNum);
    }

    List<List<String>> partitionlist = new ArrayList<>();
    public List<List<String>> partition(String s) {
        partition(s,new ArrayList<>(),0);
        return partitionlist;
    }
    public void partition(String s,List<String> list,int currentI) {
        if (currentI==s.length()){
            partitionlist.add(new ArrayList<>(list));
            return;
        }
        for (int i = currentI+1; i <= s.length(); i++) {
            String  candidate = s.substring(currentI,i);
            System.out.println(candidate);
            if (isHuiWen(candidate)){
                list.add(candidate);
                partition(s,list,i);
                list.remove(list.size()-1);
            }
        }
    }
    public boolean isHuiWen(String s){
        char[] charArray = s.toCharArray();
        int l=0;
        int r = charArray.length-1;
        while (l<r){
            if (charArray[l++]!=charArray[r--])return false;
        }
        return true;
    }

    /**
     * 用单调栈
     * @param temperatures
     * @return
     */
    public int[] dailyTemperatures(int[] temperatures) {
            int[] res = new int[temperatures.length];

            Deque<Integer> deque = new ArrayDeque<>();

        for (int i = 0; i < temperatures.length; i++) {
            int a = temperatures[i];
            while (!deque.isEmpty()){
                int topNum = deque.peekLast();
                if (a>temperatures[topNum]){
                    deque.pollLast();
                    res[topNum] = i-topNum;
                }else {
                    break;
                }
            }
            deque.offerLast(i);
        }
        return res;
    }

    public int[] nextGreaterElement(int[] nums1, int[] nums2) {
        Deque<Integer> deque = new ArrayDeque<Integer>();
        HashMap<Integer,Integer> map = new HashMap<>();
        int [] res = new int[nums1.length];
        for (int i = 0; i < nums2.length; i++) {
            int a = nums2[i];
            while (!deque.isEmpty()){
                int topNum = deque.peekLast();
                if (topNum<a){
                    deque.pollLast();
                    map.put(topNum,a);
                }else {
                    break;
                }
            }
            deque.offerLast(a);
        }
        for (int i = 0; i < res.length; i++) {
            res[i] = map.getOrDefault(nums1[i],-1);
        }
        return res;
    }


    public int[] nextGreaterElements(int[] nums) {
        int[] res = new int[nums.length];
        Arrays.fill(res,-1);
        ArrayDeque<Integer> deque = new ArrayDeque<>();
        for (int i = 0; i < nums.length*2; i++) {
            int a = nums[i%nums.length];
            while (!deque.isEmpty()){
                int topIndex = deque.peekLast();
                if (nums[topIndex]<a){
                    deque.pollLast();
                    res[topIndex] = a;
                }else break;
            }
            deque.offerLast(i%nums.length);
        }
        return res;
    }

    /**
     * 42. 接雨水
     * @param height
     * @return
     */
    public int trap(int[] height) {
        int sumOfYuShui = 0;
        Deque<Integer> deque = new ArrayDeque<>();
        for (int i = 0; i < height.length; i++) {
            int a = height[i];
            while (!deque.isEmpty()){
                int mid = deque.peekLast();
                if (height[mid]>=a){
                 break;
                }else {
                    deque.pollLast();
                    if (deque.isEmpty())continue;
                    int ll = deque.peekLast();
                    int h = Math.min(a,height[ll])-height[mid];
                    int d = i-ll;
                    int s = d*h;
                    sumOfYuShui+=s;
                }
            }
            deque.offerLast(i);
        }
        return sumOfYuShui;
    }

    /**
     *84. 柱状图中最大的矩形
     * @param heights
     * @return
     */
    public int largestRectangleArea(int[] heights) {
//        if (heights.length==1)return heights[0];
//        int maxS = Integer.MIN_VALUE;
//        Deque<Integer> deque = new ArrayDeque<>();
//        for (int i = 0; i <= heights.length; i++) {
//            int aa = -1;
//            if (i<heights.length) aa = heights[i];
//            while (!deque.isEmpty()){
//                int tt = deque.peekLast();
//                if (heights[tt]<=aa){
//                    break;
//                }else {
//                    deque.pollLast();
//                    int ll;
//                    if (deque.isEmpty())ll=-1;
//                    else ll = deque.peekLast();
//                    int width = i-ll-1;
//                    int height = heights[tt];
//                    int s = width*height;
//                    maxS = Integer.max(maxS,s);
//                }
//            }
//            deque.offerLast(i);
//        }
//        return maxS;
        int maxS = Integer.MAX_VALUE;
        Deque<Integer> deque = new ArrayDeque<>();
        for (int i = 0; i <= heights.length; i++) {
            int a;
            if (i==heights.length)a=-1;
            else a = heights[i];
            while (!deque.isEmpty()){
                int t = deque.peekLast();
                if (a>=heights[t]){
                    break;
                }else {
                    deque.pollLast();
                    int l;
                    if (deque.isEmpty())l=-1;
                    l = deque.peekLast();//左边的坐标
                    int width = i - l - 1;
                    int height = heights[t];
                    int s = width*height;
                    maxS = Math.max(maxS,s);
                }
            }
            deque.addLast(i);
        }
        return maxS;
    }

    /**
     * 整数拆分
     * @param n
     * @return
     */
    int integerBreak(int n) {
        int[] dp = new int[n+1];
        for(int i = 1; i <= n ; i++) {
            for (int j = 1; j <= i-1 ; j++) {
                int currentMax = Math.max(j*(i-j),j*dp[i-j]);
                dp[i] = Math.max(dp[i],currentMax);
            }
        }
        return dp[n];
    }

    /**
     * 416. 分割等和子集
     * 输入：nums = [1,5,11,5]
     * 输出：true
     * 解释：数组可以分割成 [1, 5, 5] 和 [11]
     * @param nums
     * @return
     */
    public boolean canPartition(int[] nums) {
        int len = nums.length;
        if (len<2)return false;
        int sum = Arrays.stream(nums).sum();
        if (sum%2!=0)return false;
        int target = sum/2;
        int[] dp = new int[target+1];
        for (int i = 0 ;i < nums.length; i++) {
            int num = nums[i];
            if (num>target)return false;
            for (int j = 0; j <= target; j++) {
                if (j>=num){
                    dp[j] = Math.max(dp[j],dp[j-num]+num);
                }
            }
        }
        if (dp[target]==target)return true;
        else return false;
    }

    /**
     * 1049. 最后一块石头的重量 II
     * @param stones
     * @return
     */
//    public int lastStoneWeightII(int[] stones) {
//        int len = stones.length;
//        int sum = Arrays.stream(stones).sum();
//        int target = sum>>1;
//        int [] dp = new int[target+1];
//        for (int i = 0; i < len; i++) {
//            int weight = stones[i];
//            for (int j = target; j >= weight; j--) {
//                dp[j] = Math.max(dp[j],dp[j-weight]+weight);
//            }
//        }
//        return sum - dp[target] - dp[target];
//    }

    /**
     * 49. 字母异位词分组
     */
    public List<List<String>> groupAnagrams(String[] strs) {
        HashMap<String,List<String>> map = new HashMap<>();
        for (String str : strs) {
            char[] chars = str.toCharArray();
            Arrays.sort(chars);
            String strKey = String.valueOf(chars);
            List<String> orDefault = map.getOrDefault(strKey, new ArrayList<String>());
            orDefault.add(str);
            map.put(strKey,orDefault);
        }
        List<List<String>> res = new ArrayList<>();
        map.forEach((key,value)->{
            res.add(value);
        });
        return res;
    }
    /**
     * 47. 全排列 II
     */
    public List<List<Integer>> quanpaixu2List = new ArrayList<>();
    public List<List<Integer>> permuteUnique(int[] nums) {
        Arrays.sort(nums);
        permuteUnique(nums,0,new ArrayList<>(),new boolean[nums.length]);
        return quanpaixu2List;
    }
    public void permuteUnique(int[] nums,int n, List<Integer> list,boolean[] booleans) {
        if (n==nums.length){
            quanpaixu2List.add(new ArrayList<>(list));
        }else {
            for (int i = 0; i < nums.length; i++) {
                if (booleans[i]||(i>0&&nums[i-1]==nums[i])&&!booleans[i-1])continue;
                //没有被操作
                booleans[i]=!booleans[i];
                list.add(nums[i]);
                permuteUnique(nums,n+1,list,booleans);
                booleans[i]=!booleans[i];
                list.remove(list.size()-1);
            }
        }
    }

    /**
     * 226. 翻转二叉树
     */
    public TreeNode invertTree(TreeNode root) {
        if(root==null)return root;
        TreeNode l = root.left;
        TreeNode r = root.right;
        TreeNode t = l;
        l = r;
        r = t;
        invertTree(l);
        invertTree(r);
        root.left = l;
        root.right = r;
        return root;
    }

    /**
     * 474. 一和零
     * @param strs
     * @param m
     * @param n
     * @return
     */
    public int findMaxForm(String[] strs, int m, int n) {
        int [][] dp = new int[m+1][n+1];
        for (String str : strs) {
            char[] chars = str.toCharArray();
            int zero = 0;
            int one = 0;
            for (char aChar : chars) {
                if (aChar-'0'==0)zero++;
                else one++;
            }
            for (int i = m; i >= zero; i--) {
                for (int j = n; j >= one; j--) {
                    dp[i][j] = Math.max(dp[i][j], dp[i - zero][j - one] + 1);
                }
            }
        }
        return dp[m][n];
    }

    /**
     * 494. 目标和
     * @param nums
     * @param target
     * @return
     */
    public int findTargetSumWays(int[] nums, int target) {
        int sum = Arrays.stream(nums).sum();
        int a = sum-target;
        if(a%2!=0||a<0)return 0;
        a=a>>1;
        int[] dp = new int[a+1];
        dp[0]=1;
        for (int num : nums) {
            for(int j = a; j>=num ;j--){
                dp[j] = dp[j]+dp[j-num];
            }
        }
        return dp[a];
    }


    /**
     * 1049. 最后一块石头的重量 II
     * @param stones
     * @return
     */
    public int lastStoneWeightII(int[] stones) {
        int sum = Arrays.stream(stones).sum();
        int target = sum+1>>1;
        int[] dp = new int[target+1];
        for (int stone : stones) {
            for (int i = target; i >= stone ; i--) {
                dp[i] = Math.max(dp[i],dp[i-stone]+stone);
            }
        }
        return sum-dp[target]*2;
    }
    //下面是之前做的
//    public int lastStoneWeightII(int[] stones) {
//        int len = stones.length;
//        int sum = Arrays.stream(stones).sum();
//        int target = sum>>1;
//        int [] dp = new int[target+1];
//        for (int i = 0; i < len; i++) {
//            int weight = stones[i];
//            for (int j = target; j >= weight; j--) {
//                dp[j] = Math.max(dp[j],dp[j-weight]+weight);
//            }
//        }
//        return sum - dp[target] - dp[target];
//    }

    /**
     *从全序与中序节点构造二叉树
     * @param preorder
     * @param inorder
     * @return
     */
//    public TreeNode buildTree(int[] preorder, int[] inorder) {
//        return buildTree(preorder, inorder,0,preorder.length-1,0,preorder.length-1);
//    }
//    public TreeNode buildTree(int[] preorder, int[] inorder,int prebegin,int preend,int inbegin, int inend) {
//        if (prebegin>preend)return null;
//        TreeNode treeNode = new TreeNode(preorder[prebegin]);
//        int targetNum = 0;
//        for (int i = 0; i < preorder.length; i++) {
//            targetNum = i;
//            if (preorder[prebegin]==inorder[i]){
//                break;
//            }
//        }
//        System.out.println(targetNum);
//        treeNode.left = buildTree(preorder, inorder,prebegin+1,prebegin+targetNum-inbegin,inbegin,targetNum-1);
//        treeNode.right = buildTree(preorder, inorder,prebegin+targetNum-inbegin+1,preend, targetNum+1, inend);
//        return treeNode;
//    }
    /**
     *279. 完全平方数
     */
    public int numSquares(int n) {
        int a = (int)Math.pow(n,0.5);
        List<Integer> nums = new ArrayList();
        for(int i=a;i>=1;i--){
            nums.add(i*i);
        }
        int []dp = new int[n+1];
        Arrays.fill(dp,Integer.MAX_VALUE-1);
        dp[0]=0;
        for(Integer num:nums){
            for(int i=num ; i<=n;i++){
                dp[i]=Math.min(dp[i],dp[i-num]+1);

            }
        }
        return dp[n];
    }

    /**
     * 322. 零钱兑换
     */
    public int coinChange(int[] coins, int amount) {
        int dp[] = new int[amount+1];
        Arrays.fill(dp,Integer.MAX_VALUE-1);
        dp[0]=0;
        for(int c : coins){
            for(int i = c; i<= amount;i++){
                dp[i]=Math.min(dp[i],dp[i-c]+1);
            }
        }
        return dp[amount]==Integer.MAX_VALUE-1?-1:dp[amount];
    }

    /**
     * 377. 组合总和 Ⅳ
     */
    public int combinationSum4(int[] nums, int target) {
        int [] dp =  new int[target+1];
        dp[0]=1;

        for(int i=0 ;i<=target;i++){

            for(int num:nums){
                if(num<=i)dp[i]=dp[i]+dp[i-num];
            }
        }
        return dp[target];
    }

    /**
     * 139. 单词拆分
     */
        public boolean wordBreak(String s, List<String> wordDict) {
            int length = s.length();
            boolean[] dp = new boolean[length +1];
            dp[0]=true;
            HashSet<String> dic = new HashSet<>(wordDict);
            for (int i = 1; i <= length; i++) {
                for (int j = 0; j < i ; j++) {
                    if (dic.contains(s.substring(j,i))&&dp[j])dp[i]=true;
                }
            }
            return dp[length];
        }

    /**
     * 213. 打家劫舍 II
     * @param nums
     * @return
     */
    public int rob(int[] nums) {
        if(nums.length==0)return 0;
        if (nums.length==1)return nums[0];
        return Math.max(rob(nums,0,nums.length-2),rob(nums,1,nums.length-1));
    }
    int rob(int[] nums,int start,int end){
        if (start==end)return nums[start];
        int dp[] = new int[nums.length];
        dp[start] = nums[start];
        dp[start+1] = Math.max(dp[start],nums[start+1]);
        for (int i = start+2; i <= end; i++) {
            dp[i] = Math.max(dp[i-1],dp[i-2]+nums[i]);
        }
        return dp[end];
    }

    /**
     * 337. 打家劫舍 III
     * @param root
     * @return
     */
    public int rob(TreeNode root) {
        int[] res = robChild(root);
        return Math.max(res[0],res[1]);
    }

    public int[] robChild(TreeNode root){
        if (root==null)return new int[]{0,0};
        int[] child1 = robChild(root.left);
        int[] child2 = robChild(root.right);
        System.out.println(root.val+Arrays.toString(child1));
        System.out.println(root.val+Arrays.toString(child2));
        int res1 = Math.max(child1[0],child1[1])+Math.max(child2[0],child2[1]);
        int res2 = root.val + child1[1] + child2[1];
        return new int[]{res2, Math.max(res1,child2[1]+child1[1])};
    }

    /**
     *121. 买卖股票的最佳时机
     */
    public int maxProfit1(int[] prices) {
        if (prices.length<=1)return 0;
        int dp[] = new int[prices.length];
        for (int i = 1; i < prices.length; i++) {
            int minus = prices[i] - prices[i - 1];
            dp[i]=minus+dp[i-1];
            if (dp[i]<0)dp[i]=0;
        }
        int res = Integer.MIN_VALUE;
        for (int i = 0; i < dp.length; i++) {
            res = Math.max(dp[i],res);
        }
        return  res;
    }

    /**
     * 123. 买卖股票的最佳时机 III
     */
//    public int maxProfit(int[] prices) {
//        if(prices.length<=1)return 0;
//        int buy1 = -prices[0], sell1 = 0;
//        int buy2 = -prices[0], sell2 = 0;
//        for (int i = 1; i < prices.length; i++) {
//            buy1 = Math.max(buy1,-prices[i]);
//            sell1 = Math.max(sell1,buy1+prices[i]);
//            buy2 = Math.max(buy2,sell1-prices[i]);
//            sell2 = Math.max(sell2,buy2+prices[i]);
//        }
//        return sell2;
//    }

    /**
     *188. 买卖股票的最佳时机 IV
     * @param k
     * @param prices
     * @return
     */
    public int maxProfit(int k, int[] prices) {
        if (prices.length<=1)return 0;
        int dp[] = new int[k*2+1];
        //初始化
        for (int i = 1; i < dp.length; i++) {
            if (i%2!=0){
                dp[i] = -prices[0];
            }else {
                dp[i] = 0;
            }
        }
        for (int i = 1; i < prices.length; i++) {
            int price = prices[i];
            for (int j = 1; j < dp.length; j++) {
                if (j%2!=0){
                    dp[j]=Math.max(dp[j-1]-price,dp[j]);//第二个下标为偶数的时候就表示为买入
                }else {
                    dp[j]=Math.max(dp[j-1]+price,dp[j]);//奇数的时候就表示为卖出
                }
            }
        }
        return dp[k*2];
    }

    /**
     * 309. 买卖股票的最佳时机含冷冻期
     * @param prices
     * @return
     */
    public int maxProfit(int[] prices) {
        int dpBuy[] = new int[prices.length];//第i天的最大的买入后的余额
        int dpSell[] = new int[prices.length];//第i天的最大的卖出后的余额
        dpBuy[0] = -prices[0];//第0天买入就是第一天价格的负数
        dpSell[0] = 0;//第0天卖出就是0
        for (int i = 1; i < prices.length; i++) {
            dpBuy[i] = Math.max(dpBuy[i-1],dpSell[Math.max(i - 2, 0)]-prices[i]);//由于卖了之后的一天内不能买，所以只能从前天推导出dpBuy,Math.max(i - 2, 0)是防止i-2<0导致数组越界
            dpSell[i] = Math.max(dpSell[i-1],dpBuy[i]+prices[i]);
        }
        return dpSell[prices.length-1];
    }

    /**
     * 106. 从中序与后序遍历序列构造二叉树
     * @param inorder
     * @param postorder
     * @return
     */
    private HashMap<Integer,Integer> nodeValIndexMap=  new HashMap<>();
    public TreeNode buildTree(int[] inorder, int[] postorder) {
        for (int i = 0; i < inorder.length; i++) {
            nodeValIndexMap.put(inorder[i],i);
        }
       return buildTree(inorder,postorder,0,inorder.length-1,0,postorder.length-1);
    }

    public TreeNode buildTree(int[] inorder, int[] postorder,int instart,int inend,int poststart,int postend){
        if (postend<poststart)return null;
        TreeNode root = new TreeNode(postorder[postend]);
        int aimIndex = nodeValIndexMap.get(postorder[postend]);
        int leftLen = aimIndex-instart;
        root.left=buildTree(inorder,postorder,instart,aimIndex-1,poststart,poststart+leftLen-1);
        root.right=buildTree(inorder,postorder,aimIndex+1,inend,poststart+leftLen,postend-1);
        return root;
    }




}








