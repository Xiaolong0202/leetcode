import java.util.*;

public class Solution {


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
        if (nums == null || nums.length == 0) return;
        int index = 0;
        //一次遍历，把非零的都往前挪
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] != 0) nums[index++] = nums[i];
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


    private int[] pricese;
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
            if (k < i) return false;
            k = Math.max(k, nums[i] + i);
        }
        return true;
    }

    public int jump(int[] nums) {
        if (nums.length == 1) return 0;
        int count = 0;
        int nextMaxWay = 0;
        int currentMaxWay = nums[0];
        for (int i = 1; i < nums.length; i++) {
            nextMaxWay = Math.max(nextMaxWay, nums[i] + i);
            if (i == currentMaxWay) {
                count++;
                currentMaxWay = nextMaxWay;
            }
        }
        return count;
    }

    public int largestSumAfterKNegations(int[] nums, int k) {
        Arrays.sort(nums);
        int i;
        int sum = 0;
        for (i = 0; i < nums.length; i++) {
            if (nums[i] < 0 && k > 0) {
                nums[i] = -nums[i];
                k--;
            }
            sum += nums[i];
        }
        if (k != 0 && k % 2 != 0) {
            Arrays.sort(nums);
            sum -= nums[0] * 2;
        }
        return sum;
    }

    /**
     * 134. 加油站
     *
     * @param gas
     * @param cost
     * @return
     */
    public int canCompleteCircuit(int[] gas, int[] cost) {
        int start = 0;
        int totalOil = 0;
        int currentOil = 0;
        for (int i = 0; i < gas.length; i++) {
            totalOil += gas[i] - cost[i];
            currentOil += gas[i] - cost[i];
            if (currentOil < 0) {
                currentOil = 0;
                start = i + 1;
            }
        }
        return totalOil >= 0 ? start : -1;
    }

    //分糖果
    public int candy(int[] ratings) {
        int len = ratings.length;
        int[] res = new int[len];//表示每个孩子分到的糖果数
        res[0] = 1;
        for (int i = 1; i < len; i++) {
            if (ratings[i] > ratings[i - 1]) {
                res[i] = res[i - 1] + 1;
            } else {
                res[i] = 1;
            }
        }
        res[len - 1] = res[len - 1];
        for (int i = len - 2; i >= 0; i--) {
            if (ratings[i] > ratings[i + 1]) res[i] = Math.max(res[i], res[i + 1] + 1);
        }
        System.out.println(Arrays.toString(res));
        int sum = 0;
        for (int i = 0; i < len; i++) {
            sum += res[i];
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
                if (o1[0] != o2[0]) {
                    return o2[0] - o1[0];
                } else {
                    return o1[1] - o2[1];
                }
            }
        });
        List<int[]> list = new ArrayList<>();
        for (int i = 0; i < people.length; i++) {
            list.add(people[i][1], people[i]);
        }
        return list.toArray(new int[people.length][2]);
    }


    public int removeElement(int[] nums, int val) {
        if (nums.length == 1 && nums[0] != val) return 1;
        if (nums.length == 0) return 0;
        int i = 0;
        int j = nums.length - 1;
        int count = 0;
        while (i < j) {
            if (nums[i] == val) {
                while (nums[j] == val && i < j) {
                    j--;
                }
                if (nums[j] == val) break;
                int temp = nums[i];
                nums[i] = nums[j];
                nums[j] = temp;
            }
            i++;
        }
        if (j == nums.length - 1 && nums[j] != val) i++;
        return i;
    }

    public int[] sortedSquares(int[] nums) {
        int[] result = new int[nums.length];
        int i;
        for (i = 0; i < nums.length; i++) {
            if (nums[i] > 0) break;
        }
        int l = i - 1;
        int r = i;
        int j = 0;
        while (l >= 0 && r <= nums.length - 1) {
            if (Math.abs(nums[l]) <= Math.abs(nums[r])) {
                result[j++] = nums[l] * nums[l];
                l--;
            } else {
                result[j++] = nums[r] * nums[r];
                r++;
            }
        }

        while (l >= 0) {
            result[j++] = nums[l] * nums[l];
            l--;
        }
        while (r <= nums.length - 1) {
            result[j++] = nums[r] * nums[r];
            r++;
        }
        return result;
    }


    public String replaceSpace(String s) {
        return s.replaceAll(" ", "%20");
    }

    public String reverseWords(String s) {
        s = s.trim();
        StringBuilder sb = new StringBuilder();
        char[] chars = s.toCharArray();
        int len = chars.length;
        int i = len - 1;
        int j = len;
        while (i >= 0) {
            if (chars[i] == ' ') {
                if ((i != len - 1) && chars[i + 1] == ' ') {
                } else {
                    sb.append(s, i + 1, j);
                    sb.append(' ');
                }
                j = i;
            } else if (i == 0) {
                sb.append(s, i, j);
            }
            i--;
        }

        return sb.toString();
    }


    List<List<Integer>> combineList = new ArrayList<>();

    public List<List<Integer>> combine(int n, int k) {
        for (int i = 1; i <= n; i++) {
            combine(1, i, n, k, new ArrayList<>());
        }
        return combineList;
    }

    public void combine(int floor, int addNum, int n, int k, List<Integer> list) {
        list.add(addNum);
        if (floor == k) {
            List<Integer> nList = new ArrayList<Integer>(list);
            combineList.add(nList);
            return;
        }
        for (int i = addNum + 1; i <= n; i++) {
            combine(floor + 1, i, n, k, list);
            list.remove(list.size() - 1);
        }
    }

    public int findMinArrowShots(int[][] points) {
        Arrays.sort(points, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return Integer.compare(o1[0], o2[0]);
            }
        });

        int right = points[0][1];
        int count = 1;

        for (int i = 1; i < points.length; i++) {
            if (points[i][0] <= right) {
                right = Math.min(points[i][1], right);
            } else {
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
                if (o1[0] == o2[0]) return Integer.compare(o1[1], o2[1]);
                else return Integer.compare(o1[0], o2[0]);
            }
        });

        int right = intervals[0][1];
        int count = 0;
        for (int i = 1; i < intervals.length; i++) {
            if (intervals[i][0] >= right) {
                // System.out.println(right+" "+intervals[i][1]);
                right = intervals[i][1];
            } else {
                count++;
                right = Math.min(right, intervals[i][1]);
            }
        }
        return count;
    }

    public List<Integer> partitionLabels(String s) {
        char[] chars = s.toCharArray();
        List<Integer> list = new ArrayList<>();
        HashMap<Character, Integer> map = new HashMap<>();
        for (int i = 0; i < chars.length; i++) {
            map.put(chars[i], i);
        }
        int preR = -1;
        int right = map.get(chars[0]);
        for (int i = 1; i < chars.length; i++) {
            int temp = map.get(chars[i]);
            if (i > right) {
                list.add(right - preR);
                preR = right;
            }
            right = Math.max(right, temp);
        }
        list.add(right - preR);
        return list;
    }

    public int[][] merge(int[][] intervals) {
        Arrays.sort(intervals, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                if (o1[0] != o2[0]) return Integer.compare(o1[0], o2[0]);
                else return Integer.compare(o1[1], o2[1]);
            }
        });
        List<int[]> list = new ArrayList<>();
        int left = intervals[0][0];
        int right = intervals[0][1];

        for (int i = 1; i < intervals.length; i++) {
            if (intervals[i][0] <= right) {
                right = Math.max(intervals[i][1], right);
            } else {
                int[] a = new int[]{left, right};
                list.add(a);
                left = intervals[i][0];
                right = intervals[i][1];
            }
        }
        list.add(new int[]{left, right});
        int[][] ints = list.toArray(new int[list.size()][2]);
        return ints;
    }

    public int monotoneIncreasingDigits(int n) {
        String string = Integer.toString(n);
        char[] num = string.toCharArray();
        for (int i = 0; i < num.length - 1; i++) {
            if (num[i] > num[i + 1]) {
                while (i > 0 && num[i] == num[i - 1]) {
                    i--;
                }
                num[i++]++;
                while (i < num.length) {
                    num[i++] = '9';
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
        combinationSum3(k, n, 0, 0, 0, new ArrayList<>());
        return combinationListSum3;
    }

    public void combinationSum3(int k, int n, int floor, int currentNum, int sum, List<Integer> list) {
        if (sum > n) {
        } else if (sum == n && k < floor) {
        } else if (k == floor) {
            if (sum == n) combinationListSum3.add(new ArrayList<>(list));
        } else {
            int maxI = n > 9 ? 9 : n - 1;
            for (int i = currentNum + 1; i <= maxI; i++) {
                list.add(i);
                combinationSum3(k, n, floor + 1, i, i + sum, list);
                list.remove(list.size() - 1);
            }
        }
    }

    List<List<Integer>> subsetDupList = new ArrayList<>();

    public List<List<Integer>> subsetsWithDup(int[] nums) {
        Arrays.sort(nums);
        boolean[][] subsetArr = new boolean[11][21];
        List<Integer> list = new ArrayList<>();
        subsetDupList.add(list);
        subsetsWithDup(nums, list, -1, 0, subsetArr);
        return subsetDupList;
    }

    public void subsetsWithDup(int[] nums, List<Integer> list, int i, int floor, boolean[][] subsetArr) {
        if (floor == nums.length) return;
        for (int j = i + 1; j < nums.length; j++) {
            if (!subsetArr[floor][nums[j] + 10]) {
                list.add(nums[j]);
                subsetDupList.add(new ArrayList<>(list));
                subsetArr[floor][nums[j] + 10] = true;
                subsetsWithDup(nums, list, j, floor + 1, subsetArr);
                list.remove(list.size() - 1);
//                subsetArr[floor][nums[j]]=false;
            }
        }
        Arrays.fill(subsetArr[floor], false);
    }

    List<List<Integer>> combinationList = new ArrayList<>();

    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        Arrays.sort(candidates);
        combinationSum(candidates, target, -1, 0, 0, new ArrayList<>());
        return combinationList;
    }

    public void combinationSum(int[] candidates, int target, int i, int floor, int currentSum, List<Integer> list) {
        if (currentSum > target) return;
        if (currentSum == target) {
            combinationList.add(new ArrayList<>(list));
            return;
        }
        for (int j = i + 1; j < candidates.length; j++) {
            int countPlusTime = 0;
            int a = candidates[j] + currentSum;
            while (a <= target) {
                list.add(candidates[j]);
                combinationSum(candidates, target, j, floor + 1, a, list);
                countPlusTime++;
                a += candidates[j];
            }
            for (int k = 0; k < countPlusTime; k++) {
                list.remove(list.size() - 1);
            }
        }
    }

    public int[][] generateMatrix(int n) {
        int[][] res = new int[n][n];
        generateMatrix(0, n, res, 1);
        return res;
    }

    public void generateMatrix(int m, int n, int[][] res, int currentNum) {
        for (int i = m; i < n - m; i++) {
            res[m][i] = currentNum++;
        }

        for (int i = m + 1; i < n - m; i++) {
            res[i][n - m - 1] = currentNum++;
        }

        for (int i = n - m - 2; i >= m; i--) {
            res[n - m - 1][i] = currentNum++;
        }

        for (int i = n - m - 2; i > m; i--) {
            res[i][m] = currentNum++;
        }
        if (currentNum == n * n + 1) return;
        generateMatrix(m + 1, n, res, currentNum);
    }

    List<List<String>> partitionlist = new ArrayList<>();

    public List<List<String>> partition(String s) {
        partition(s, new ArrayList<>(), 0);
        return partitionlist;
    }

    public void partition(String s, List<String> list, int currentI) {
        if (currentI == s.length()) {
            partitionlist.add(new ArrayList<>(list));
            return;
        }
        for (int i = currentI + 1; i <= s.length(); i++) {
            String candidate = s.substring(currentI, i);
            System.out.println(candidate);
            if (isHuiWen(candidate)) {
                list.add(candidate);
                partition(s, list, i);
                list.remove(list.size() - 1);
            }
        }
    }

    public boolean isHuiWen(String s) {
        char[] charArray = s.toCharArray();
        int l = 0;
        int r = charArray.length - 1;
        while (l < r) {
            if (charArray[l++] != charArray[r--]) return false;
        }
        return true;
    }

    /**
     * 用单调栈
     *
     * @param temperatures
     * @return
     */
    public int[] dailyTemperatures(int[] temperatures) {
        int[] res = new int[temperatures.length];

        Deque<Integer> deque = new ArrayDeque<>();

        for (int i = 0; i < temperatures.length; i++) {
            int a = temperatures[i];
            while (!deque.isEmpty()) {
                int topNum = deque.peekLast();
                if (a > temperatures[topNum]) {
                    deque.pollLast();
                    res[topNum] = i - topNum;
                } else {
                    break;
                }
            }
            deque.offerLast(i);
        }
        return res;
    }

    public int[] nextGreaterElement(int[] nums1, int[] nums2) {
        Deque<Integer> deque = new ArrayDeque<Integer>();
        HashMap<Integer, Integer> map = new HashMap<>();
        int[] res = new int[nums1.length];
        for (int i = 0; i < nums2.length; i++) {
            int a = nums2[i];
            while (!deque.isEmpty()) {
                int topNum = deque.peekLast();
                if (topNum < a) {
                    deque.pollLast();
                    map.put(topNum, a);
                } else {
                    break;
                }
            }
            deque.offerLast(a);
        }
        for (int i = 0; i < res.length; i++) {
            res[i] = map.getOrDefault(nums1[i], -1);
        }
        return res;
    }


    public int[] nextGreaterElements(int[] nums) {
        int[] res = new int[nums.length];
        Arrays.fill(res, -1);
        ArrayDeque<Integer> deque = new ArrayDeque<>();
        for (int i = 0; i < nums.length * 2; i++) {
            int a = nums[i % nums.length];
            while (!deque.isEmpty()) {
                int topIndex = deque.peekLast();
                if (nums[topIndex] < a) {
                    deque.pollLast();
                    res[topIndex] = a;
                } else break;
            }
            deque.offerLast(i % nums.length);
        }
        return res;
    }

    /**
     * 42. 接雨水
     *
     * @param height
     * @return
     */
    public int trap(int[] height) {
        int sumOfYuShui = 0;
        Deque<Integer> deque = new ArrayDeque<>();
        for (int i = 0; i < height.length; i++) {
            int a = height[i];
            while (!deque.isEmpty()) {
                int mid = deque.peekLast();
                if (height[mid] >= a) {
                    break;
                } else {
                    deque.pollLast();
                    if (deque.isEmpty()) continue;
                    int ll = deque.peekLast();
                    int h = Math.min(a, height[ll]) - height[mid];
                    int d = i - ll;
                    int s = d * h;
                    sumOfYuShui += s;
                }
            }
            deque.offerLast(i);
        }
        return sumOfYuShui;
    }

    /**
     * 84. 柱状图中最大的矩形
     *
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
            if (i == heights.length) a = -1;
            else a = heights[i];
            while (!deque.isEmpty()) {
                int t = deque.peekLast();
                if (a >= heights[t]) {
                    break;
                } else {
                    deque.pollLast();
                    int l;
                    if (deque.isEmpty()) l = -1;
                    l = deque.peekLast();//左边的坐标
                    int width = i - l - 1;
                    int height = heights[t];
                    int s = width * height;
                    maxS = Math.max(maxS, s);
                }
            }
            deque.addLast(i);
        }
        return maxS;
    }

    /**
     * 整数拆分
     *
     * @param n
     * @return
     */
    int integerBreak(int n) {
        int[] dp = new int[n + 1];
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= i - 1; j++) {
                int currentMax = Math.max(j * (i - j), j * dp[i - j]);
                dp[i] = Math.max(dp[i], currentMax);
            }
        }
        return dp[n];
    }

    /**
     * 416. 分割等和子集
     * 输入：nums = [1,5,11,5]
     * 输出：true
     * 解释：数组可以分割成 [1, 5, 5] 和 [11]
     *
     * @param nums
     * @return
     */
    public boolean canPartition(int[] nums) {
        int len = nums.length;
        if (len < 2) return false;
        int sum = Arrays.stream(nums).sum();
        if (sum % 2 != 0) return false;
        int target = sum / 2;
        int[] dp = new int[target + 1];
        for (int i = 0; i < nums.length; i++) {
            int num = nums[i];
            if (num > target) return false;
            for (int j = 0; j <= target; j++) {
                if (j >= num) {
                    dp[j] = Math.max(dp[j], dp[j - num] + num);
                }
            }
        }
        return dp[target] == target;
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
        HashMap<String, List<String>> map = new HashMap<>();
        for (String str : strs) {
            char[] chars = str.toCharArray();
            Arrays.sort(chars);
            String strKey = String.valueOf(chars);
            List<String> orDefault = map.getOrDefault(strKey, new ArrayList<String>());
            orDefault.add(str);
            map.put(strKey, orDefault);
        }
        List<List<String>> res = new ArrayList<>();
        map.forEach((key, value) -> {
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
        permuteUnique(nums, 0, new ArrayList<>(), new boolean[nums.length]);
        return quanpaixu2List;
    }

    public void permuteUnique(int[] nums, int n, List<Integer> list, boolean[] booleans) {
        if (n == nums.length) {
            quanpaixu2List.add(new ArrayList<>(list));
        } else {
            for (int i = 0; i < nums.length; i++) {
                if (booleans[i] || (i > 0 && nums[i - 1] == nums[i]) && !booleans[i - 1]) continue;
                //没有被操作
                booleans[i] = !booleans[i];
                list.add(nums[i]);
                permuteUnique(nums, n + 1, list, booleans);
                booleans[i] = !booleans[i];
                list.remove(list.size() - 1);
            }
        }
    }

    /**
     * 226. 翻转二叉树
     */
    public TreeNode invertTree(TreeNode root) {
        if (root == null) return root;
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
     *
     * @param strs
     * @param m
     * @param n
     * @return
     */
    public int findMaxForm(String[] strs, int m, int n) {
        int[][] dp = new int[m + 1][n + 1];
        for (String str : strs) {
            char[] chars = str.toCharArray();
            int zero = 0;
            int one = 0;
            for (char aChar : chars) {
                if (aChar - '0' == 0) zero++;
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
     *
     * @param nums
     * @param target
     * @return
     */
    public int findTargetSumWays(int[] nums, int target) {
        int sum = Arrays.stream(nums).sum();
        int a = sum - target;
        if (a % 2 != 0 || a < 0) return 0;
        a = a >> 1;
        int[] dp = new int[a + 1];
        dp[0] = 1;
        for (int num : nums) {
            for (int j = a; j >= num; j--) {
                dp[j] = dp[j] + dp[j - num];
            }
        }
        return dp[a];
    }


    /**
     * 1049. 最后一块石头的重量 II
     *
     * @param stones
     * @return
     */
    public int lastStoneWeightII(int[] stones) {
        int sum = Arrays.stream(stones).sum();
        int target = sum + 1 >> 1;
        int[] dp = new int[target + 1];
        for (int stone : stones) {
            for (int i = target; i >= stone; i--) {
                dp[i] = Math.max(dp[i], dp[i - stone] + stone);
            }
        }
        return sum - dp[target] * 2;
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
     * 279. 完全平方数
     */
    public int numSquares(int n) {
        int a = (int) Math.pow(n, 0.5);
        List<Integer> nums = new ArrayList();
        for (int i = a; i >= 1; i--) {
            nums.add(i * i);
        }
        int[] dp = new int[n + 1];
        Arrays.fill(dp, Integer.MAX_VALUE - 1);
        dp[0] = 0;
        for (Integer num : nums) {
            for (int i = num; i <= n; i++) {
                dp[i] = Math.min(dp[i], dp[i - num] + 1);

            }
        }
        return dp[n];
    }

    /**
     * 322. 零钱兑换
     */
    public int coinChange(int[] coins, int amount) {
        int[] dp = new int[amount + 1];
        Arrays.fill(dp, Integer.MAX_VALUE - 1);
        dp[0] = 0;
        for (int c : coins) {
            for (int i = c; i <= amount; i++) {
                dp[i] = Math.min(dp[i], dp[i - c] + 1);
            }
        }
        return dp[amount] == Integer.MAX_VALUE - 1 ? -1 : dp[amount];
    }

    /**
     * 377. 组合总和 Ⅳ
     */
    public int combinationSum4(int[] nums, int target) {
        int[] dp = new int[target + 1];
        dp[0] = 1;

        for (int i = 0; i <= target; i++) {

            for (int num : nums) {
                if (num <= i) dp[i] = dp[i] + dp[i - num];
            }
        }
        return dp[target];
    }

    /**
     * 139. 单词拆分
     */
    public boolean wordBreak(String s, List<String> wordDict) {
        int length = s.length();
        boolean[] dp = new boolean[length + 1];
        dp[0] = true;
        HashSet<String> dic = new HashSet<>(wordDict);
        for (int i = 1; i <= length; i++) {
            for (int j = 0; j < i; j++) {
                if (dic.contains(s.substring(j, i)) && dp[j]) {
                    dp[i] = true;
                    break;
                }
            }
        }
        return dp[length];
    }

    /**
     * 213. 打家劫舍 II
     *
     * @param nums
     * @return
     */
    public int rob(int[] nums) {
        if (nums.length == 0) return 0;
        if (nums.length == 1) return nums[0];
        return Math.max(rob(nums, 0, nums.length - 2), rob(nums, 1, nums.length - 1));
    }

    int rob(int[] nums, int start, int end) {
        if (start == end) return nums[start];
        int[] dp = new int[nums.length];
        dp[start] = nums[start];
        dp[start + 1] = Math.max(dp[start], nums[start + 1]);
        for (int i = start + 2; i <= end; i++) {
            dp[i] = Math.max(dp[i - 1], dp[i - 2] + nums[i]);
        }
        return dp[end];
    }

    /**
     * 337. 打家劫舍 III
     *
     * @param root
     * @return
     */
    public int rob(TreeNode root) {
        int[] res = robChild(root);
        return Math.max(res[0], res[1]);
    }

    public int[] robChild(TreeNode root) {
        if (root == null) return new int[]{0, 0};
        int[] child1 = robChild(root.left);
        int[] child2 = robChild(root.right);
        System.out.println(root.val + Arrays.toString(child1));
        System.out.println(root.val + Arrays.toString(child2));
        int res1 = Math.max(child1[0], child1[1]) + Math.max(child2[0], child2[1]);
        int res2 = root.val + child1[1] + child2[1];
        return new int[]{res2, Math.max(res1, child2[1] + child1[1])};
    }

    /**
     * 121. 买卖股票的最佳时机
     */
    public int maxProfit1(int[] prices) {
        if (prices.length <= 1) return 0;
        int[] dp = new int[prices.length];
        for (int i = 1; i < prices.length; i++) {
            int minus = prices[i] - prices[i - 1];
            dp[i] = minus + dp[i - 1];
            if (dp[i] < 0) dp[i] = 0;
        }
        int res = Integer.MIN_VALUE;
        for (int i = 0; i < dp.length; i++) {
            res = Math.max(dp[i], res);
        }
        return res;
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
     * 188. 买卖股票的最佳时机 IV
     *
     * @param k
     * @param prices
     * @return
     */
    public int maxProfit(int k, int[] prices) {
        if (prices.length <= 1) return 0;
        int[] dp = new int[k * 2 + 1];
        //初始化
        for (int i = 1; i < dp.length; i++) {
            if (i % 2 != 0) {
                dp[i] = -prices[0];
            } else {
                dp[i] = 0;
            }
        }
        for (int i = 1; i < prices.length; i++) {
            int price = prices[i];
            for (int j = 1; j < dp.length; j++) {
                if (j % 2 != 0) {
                    dp[j] = Math.max(dp[j - 1] - price, dp[j]);//第二个下标为偶数的时候就表示为买入
                } else {
                    dp[j] = Math.max(dp[j - 1] + price, dp[j]);//奇数的时候就表示为卖出
                }
            }
        }
        return dp[k * 2];
    }

    /**
     * 309. 买卖股票的最佳时机含冷冻期
     *
     * @param prices
     * @return
     */
    public int maxProfit(int[] prices) {
        int[] dpBuy = new int[prices.length];//第i天的最大的买入后的余额
        int[] dpSell = new int[prices.length];//第i天的最大的卖出后的余额
        dpBuy[0] = -prices[0];//第0天买入就是第一天价格的负数
        dpSell[0] = 0;//第0天卖出就是0
        for (int i = 1; i < prices.length; i++) {
            dpBuy[i] = Math.max(dpBuy[i - 1], dpSell[Math.max(i - 2, 0)] - prices[i]);//由于卖了之后的一天内不能买，所以只能从前天推导出dpBuy,Math.max(i - 2, 0)是防止i-2<0导致数组越界
            dpSell[i] = Math.max(dpSell[i - 1], dpBuy[i] + prices[i]);
        }
        return dpSell[prices.length - 1];
    }

    /**
     * 106. 从中序与后序遍历序列构造二叉树
     *
     * @param inorder
     * @param postorder
     * @return
     */
    private final HashMap<Integer, Integer> nodeValIndexMap = new HashMap<>();

    public TreeNode buildTree(int[] inorder, int[] postorder) {
        for (int i = 0; i < inorder.length; i++) {
            nodeValIndexMap.put(inorder[i], i);
        }
        return buildTree(inorder, postorder, 0, inorder.length - 1, 0, postorder.length - 1);
    }

    public TreeNode buildTree(int[] inorder, int[] postorder, int instart, int inend, int poststart, int postend) {
        if (postend < poststart) return null;
        TreeNode root = new TreeNode(postorder[postend]);
        int aimIndex = nodeValIndexMap.get(postorder[postend]);
        int leftLen = aimIndex - instart;
        root.left = buildTree(inorder, postorder, instart, aimIndex - 1, poststart, poststart + leftLen - 1);
        root.right = buildTree(inorder, postorder, aimIndex + 1, inend, poststart + leftLen, postend - 1);
        return root;
    }


    /**
     * 714. 买卖股票的最佳时机含手续费
     *
     * @param prices
     * @param fee
     * @return
     */
    public int maxProfit(int[] prices, int fee) {
        int[] dpBuy = new int[prices.length];
        int[] dpSell = new int[prices.length];
        dpBuy[0] = -prices[0];
        for (int i = 1; i < prices.length; i++) {
            dpBuy[i] = Math.max(dpBuy[i - 1], dpSell[i - 1] - prices[i]);
            dpSell[i] = Math.max(dpSell[i - 1], dpBuy[i] + prices[i] - fee);
        }
        return dpSell[prices.length - 1];
    }

    /**
     * 718. 最长重复子数组
     *
     * @param nums1
     * @param nums2
     * @return
     */
    //动态规划
    public int findLength(int[] nums1, int[] nums2) {
        int res = Integer.MIN_VALUE;
        int[][] dp = new int[nums1.length + 1][nums2.length + 1];
        for (int i = 1; i <= nums2.length; i++) {
            for (int j = 1; j <= nums1.length; j++) {
                if (nums2[i - 1] == nums1[j - 1]) {
                    dp[j][i] = dp[j - 1][i - 1] + 1;
                }
                res = Math.max(res, dp[j][i]);
            }
        }
        return res;
    }

    /**
     * 1143. 最长公共子序列
     */
    public int longestCommonSubsequence(String text1, String text2) {
        char[] charArr1 = text1.toCharArray();
        char[] charArr2 = text2.toCharArray();

        int[][] dp = new int[charArr1.length + 1][charArr2.length + 1];
        for (int i = 1; i <= charArr1.length; i++) {
            for (int j = 1; j <= charArr2.length; j++) {
                if (charArr1[i - 1] == charArr2[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
                //    dp[i][j]=Math.max(dp[i][j],Math.max(dp[i-1][j],dp[i][j-1]));
            }
        }
        return dp[charArr1.length][charArr2.length];
    }

    /**
     * 1035. 不相交的线
     *
     * @param nums1
     * @param nums2
     * @return
     */
    public int maxUncrossedLines(int[] nums1, int[] nums2) {
        int[][] dp = new int[nums1.length + 1][nums2.length + 1];
        for (int i = 1; i <= nums1.length; i++) {
            for (int j = 1; j <= nums2.length; j++) {
                if (nums1[i - 1] == nums2[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[nums1.length][nums2.length];
    }

    /**
     * 392. 判断子序列
     *
     * @param s
     * @param t
     * @return
     */
    public boolean isSubsequence(String s, String t) {
        char[] charArrays = s.toCharArray();
        char[] charArrayt = t.toCharArray();
        int i = 0;
        int j = 0;
        while (i < charArrayt.length && j < charArrays.length) {
            if (charArrays[j] == charArrayt[i]) {
                j++;
                i++;
            } else i++;
        }
        return j == charArrays.length;
    }

    /**
     * 115. 不同的子序列
     *
     * @param s
     * @param t
     * @return
     */
    public int numDistinct(String s, String t) {
        char[] sCharArray = s.toCharArray();
        char[] tCharArray = t.toCharArray();
        int[][] dp = new int[sCharArray.length + 1][tCharArray.length + 1];
        for (int i = 0; i < dp.length; i++) {
            dp[i][0] = 1;
        }
        for (int i = 1; i <= sCharArray.length; i++) {
            for (int j = 1; j <= tCharArray.length; j++) {
                if (sCharArray[i - 1] == tCharArray[j - 1]) {
                    dp[i][j] = dp[i - 1][j] + dp[i - 1][j - 1];
                } else {
                    dp[i][j] = dp[i - 1][j];
                }
            }
        }
        return dp[sCharArray.length][tCharArray.length];
    }

    /**
     * 617. 合并二叉树
     *
     * @param root1
     * @param root2
     * @return
     */
    public TreeNode mergeTrees(TreeNode root1, TreeNode root2) {
        if (root1 == null) {
            return root2;
        }
        if (root2 == null) {
            return root1;
        }
        TreeNode treeNode = new TreeNode();
        treeNode.val = root1.val + root2.val;
        treeNode.left = mergeTrees(root1.left, root2.left);
        treeNode.right = mergeTrees(root1.right, root2.right);
        return treeNode;
    }


    /**
     * 96. 不同的二叉搜索树
     *
     * @param n
     * @return
     */
    public int numTrees(int n) {
        int[] dp = new int[n + 1];
        dp[0] = 1;
        dp[1] = 1;
        for (int i = 2; i <= n; i++) {
            for (int j = 0; j <= i - 1; j++) {
                dp[i] += dp[j] * dp[i - 1 - j];
            }
        }
        return dp[n];
    }

    public int longestConsecutive(int[] nums) {
        if (nums.length == 0) return 0;
        Arrays.sort(nums);
        int maxLen = 1;
        int len = 1;
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] - 1 == nums[i - 1]) {
                len++;
            } else if (nums[i] == nums[i - 1]) {

            } else {
                len = 1;
            }
            maxLen = Math.max(len, maxLen);
        }
        return maxLen;
    }

    /**
     * 114. 二叉树展开为链表
     *
     * @param root
     */
    public void flatten(TreeNode root) {
        //迭代法
//        if (root == null) return;
//        TreeNode head = new TreeNode();
//        TreeNode p = head;
//        Deque<TreeNode> deque = new ArrayDeque<>();
//        deque.offerLast(root);
//        while (!deque.isEmpty()){
//            TreeNode node = deque.pollLast();
//            //将结点添加至链表当中
//            TreeNode listNode = new TreeNode();
//            System.out.println("node.val = " + node.val);
//            listNode.val = node.val;
//            p.right = listNode;
//            p = p.right;
//            //
//            if (node.right!=null)deque.offerLast(node.right);
//            if (node.left!=null)deque.offerLast(node.left);
//        }
//            head = head.right;
//            root.left = null;
//            root.val = head.val;
//            root.right = head.right;
        //方法2
        while (root != null) {
            if (root.left != null) {
                TreeNode left = root.left;
                while (left.right != null) {
                    left = left.right;
                }
                left.right = root.right;
                root.right = root.left;
            }
            root.left = null;
            root = root.right;
        }
    }

    /**
     * 78. 子集
     *
     * @param nums
     * @return
     */
    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> container = new ArrayList<>();
        container.add(new ArrayList<>());
        subsets(nums, -1, 0, new ArrayList<>(), container);
        return container;
    }

    public void subsets(int[] nums, int i, int n, List<Integer> list, List<List<Integer>> container) {
        if (n >= nums.length) return;
        for (int j = i + 1; j < nums.length; j++) {
            list.add(nums[j]);
            container.add(new ArrayList<>(list));
            subsets(nums, j, n + 1, list, container);
            list.remove(list.size() - 1);
        }
    }

    /**
     * 23. 合并 K 个升序链表
     *
     * @param lists
     * @return
     */
    public ListNode mergeKLists(ListNode[] lists) {
        ListNode head = new ListNode();
        ListNode res = head;
        PriorityQueue<ListNode> priorityQueue = new PriorityQueue<>(Comparator.comparingInt(o -> o.val));
        for (ListNode node : lists) {
            while (node != null) {
                priorityQueue.add(node);
                node = node.next;
            }
        }
        while (!priorityQueue.isEmpty()) {
            ListNode poll = priorityQueue.poll();
            head.next = poll;
            head = head.next;
            head.next = null;
        }
        return res.next;
//        if(lists.length<=0)return null;
//
//        ListNode head = new ListNode();
//        ListNode returnHead = head;
//        int endCount = 0;
//
//        while (true) {
//            int p = 0;
////            int shangyige = 0;
//            //选出最小的结点
//            for (int i = 0; i < lists.length; i++) {
//                if (lists[i] == null) continue;
//                if (lists[p]==null){
//                    p = i;
//                }
//                if (lists[p].val >= lists[i].val) {
////                    shangyige = p;
//                    p = i;
//                }
//            }
//            if (lists[p]!=null){
//                ListNode addedNode = new ListNode();
//                addedNode.val =  lists[p].val;
//                //将addedNode添加
//                head.next = addedNode;
//                head = head.next;
//                //
//                lists[p] = lists[p].next;
//            }
//            //如果被添加的结点为最后一个结点
//            if (lists[p]==null){
//                endCount++;
////                p = shangyige;
//            }
//
//            if (endCount >= lists.length) break;
//        }
//        return returnHead.next;
    }

    /**
     * 148. 排序链表
     *
     * @param head
     * @return
     */
    public ListNode sortList(ListNode head) {
        if (head == null || head.next == null) return head;
        ListNode slow = head;
        ListNode fast = head.next;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        ListNode right = sortList(slow.next);
        slow.next = null;
        ListNode left = sortList(head);
        ListNode res = new ListNode();
        ListNode R = res;
        while (left != null && right != null) {
            if (left.val < right.val) {
                res.next = left;
                left = left.next;
            } else {
                res.next = right;
                right = right.next;
            }
            res = res.next;
            res.next = null;
        }
        if (left != null) {
            res.next = left;
        } else if (right != null) {
            res.next = right;
        }
        return R.next;
    }

    /**
     * 160. 相交链表
     *
     * @param headA
     * @param headB
     * @return
     */
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        ListNode a = headA;
        ListNode b = headB;
        while (a != b) {
            a = a == null ? headB : a.next;
            b = b == null ? headA : b.next;
        }
        return a;
    }


    /**
     * 142. 环形链表 II
     *
     * @param head
     * @return
     */
    public ListNode detectCycle(ListNode head) {
        if (head == null || head.next == null) return null;
        ListNode slow = head;
        ListNode fast = head;
        while (true) {
            if (fast == null || fast.next == null) return null;
            fast = fast.next.next;
            slow = slow.next;
            if (fast == slow) break;
        }
        fast = head;
        while (fast != slow) {
            fast = fast.next;
            slow = slow.next;
        }
        return fast;
    }

    /**
     * 46. 全排列
     *
     * @param nums
     * @return
     */
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        permute(nums, new int[nums.length], res, new ArrayList<>());
        return res;
    }


    public void permute(int[] nums, int[] visited, List<List<Integer>> res, List<Integer> list) {
        if (list.size() >= nums.length) {
            res.add(new ArrayList<>(list));
            return;
        }
        for (int j = 0; j < nums.length; j++) {
            if (visited[j] == 0) {
                visited[j] = 1;
                list.add(nums[j]);
                permute(nums, visited, res, list);
                visited[j] = 0;
                list.remove(list.size() - 1);
            }
        }
    }

    /**
     * 169. 多数元素 投票法求质数
     *
     * @param nums
     * @return
     */
    public int majorityElement(int[] nums) {
        int count = 0;
        int candidate = 0;
        for (int num : nums) {
            if (count == 0) candidate = num;
            if (num != candidate) count--;
            else count++;
        }
        return candidate;
    }

    /**
     * 200. 岛屿数量
     *
     * @param grid
     * @return
     */
    public int numIslands(char[][] grid) {
        if (grid.length <= 0) return 0;
        boolean[][] flag = new boolean[grid.length][grid[0].length];
        int res = 0;
        int[] count = new int[]{0};
        int total = grid.length * grid[0].length;
        ok:
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[i].length; j++) {
                if (!flag[i][j]) {
                    if (grid[i][j] == '1') {
                        bfsNumsIsland(grid, flag, i, j, count);
                        res++;
                    } else {
                        flag[i][j] = true;
                        count[0]++;
                    }
                }
                if (count[0] >= total) break ok;
            }
        }
        return res;
    }

    private void bfsNumsIsland(char[][] grid, boolean[][] flag, int i, int j, int[] count) {
        Deque<int[]> deque = new ArrayDeque<>();
        deque.offerLast(new int[]{i, j});
        while (!deque.isEmpty()) {
            int[] location = deque.pollFirst();
            int x = location[0];
            int y = location[1];
            if (!flag[x][y] && grid[x][y] == '1') {
                flag[x][y] = true;//标记
                count[0]++;
                if (x + 1 < grid.length) {
                    deque.offerLast(new int[]{x + 1, y});
                }
                if (y + 1 < grid[0].length) {
                    deque.offerLast(new int[]{x, y + 1});
                }
                if (x - 1 >= 0) {
                    deque.offerLast(new int[]{x - 1, y});
                }
                if (y - 1 >= 0) {
                    deque.offerLast(new int[]{x, y - 1});
                }
            }
        }
    }

    /**
     * 208. 实现 Trie (前缀树)
     */
    class Trie {

        class TrieNode {
            Character c;//value
            Map<Character, TrieNode> trieNodeMap;

            public TrieNode(Character c) {
                // System.out.println("System.getProperty(\"java.version\") = " + System.getProperty("java.version"));
                this.c = c;
                trieNodeMap = new HashMap<>();
            }

            public void insert(char[] chars, int index) {
                if (index >= chars.length) return;
                char aChar = chars[index];
                if (!trieNodeMap.containsKey(aChar)) {
                    trieNodeMap.put(aChar, new TrieNode(aChar));
                }
                trieNodeMap.get(aChar).insert(chars, index + 1);
            }

            public boolean find(char[] chars, int index) {
                if (index >= chars.length) return true;
                char aChar = chars[index];
                if (trieNodeMap.containsKey(aChar)) {
                    return trieNodeMap.get(aChar).find(chars, index + 1);
                } else {
                    return false;
                }
            }
        }

        private final TrieNode root;
        private final HashSet<String> set;

        public Trie() {
            root = new TrieNode(null);
            set = new HashSet<>();
        }

        public void insert(String word) {
            set.add(word);
            root.insert(word.toCharArray(), 0);
        }

        public boolean search(String word) {
            return set.contains(word);
        }

        public boolean startsWith(String prefix) {
            return root.find(prefix.toCharArray(), 0);
        }
    }

    /**
     * 13. 罗马数字转整数
     *
     * @param s
     * @return
     */
    public int romanToInt(String s) {
        Map<Character, Integer> map = new HashMap<>();
        map.put('I', 1);
        map.put('V', 5);
        map.put('X', 10);
        map.put('L', 50);
        map.put('C', 100);
        map.put('D', 500);
        map.put('M', 1000);

        char[] charArray = s.toCharArray();
        int res = 0;
        int tempSum = 0;
        for (int i = 0; i < charArray.length; i++) {
            if (i + 1 >= charArray.length) {
                res += map.get(charArray[i]);
                break;
            }
            char currentC = charArray[i];
            char nextC = charArray[i + 1];
            Integer currentV = map.get(currentC);
            Integer nextV = map.get(nextC);
            if (currentV >= nextV) {
                res += currentV;
            } else {
                res -= currentV;
            }
        }
        return res;
    }

    /**
     * 543. 二叉树的直径
     *
     * @param root
     * @return
     */
    private int diameterOfBinaryTreeRes = Integer.MIN_VALUE;

    public int diameterOfBinaryTree(TreeNode root) {
        maxBinaryTreeDepth(root);
        return diameterOfBinaryTreeRes;
    }

    public int maxBinaryTreeDepth(TreeNode root) {
        if (root == null) return -1;
        else {
            int leftMaxLen = maxBinaryTreeDepth(root.left);
            int rightMaxLen = maxBinaryTreeDepth(root.right);
            leftMaxLen++;
            rightMaxLen++;
            //看该结点的哪两个腿最长
            diameterOfBinaryTreeRes = Math.max(diameterOfBinaryTreeRes, leftMaxLen + rightMaxLen);
            return Math.max(leftMaxLen, rightMaxLen);
        }
    }


    /**
     * 199. 二叉树的右视图
     *
     * @param root
     * @return
     */
    public List<Integer> rightSideView(TreeNode root) {
        int[] rightSideViewFloorVisited = new int[101];
        List<Integer> list = new ArrayList<>();
        rightSideView(root, 0, rightSideViewFloorVisited, list);
        return list;
    }

    public void rightSideView(TreeNode root, int floor, int[] rightSideViewFloorVisited, List<Integer> res) {
        if (root == null) return;
        //某一层之前没有被访问过，就直接放入List当中
        if (rightSideViewFloorVisited[floor] == 0) {
            res.add(root.val);
            rightSideViewFloorVisited[floor] = 1;
        }
        rightSideView(root.right, floor + 1, rightSideViewFloorVisited, res);
        rightSideView(root.left, floor + 1, rightSideViewFloorVisited, res);
    }

    /**
     * 230. 二叉搜索树中第K小的元素
     *
     * @param root
     * @param k
     * @return
     */
    public int kthSmallest(TreeNode root, int k) {
        Deque<TreeNode> deque = new ArrayDeque();
        while (!deque.isEmpty() || root != null) {
            while (root != null) {
                deque.offerLast(root);
                root = root.left;
            }
            root = deque.pollLast();
            if (--k <= 0) return root.val;
            root = root.right;
        }
        return -1;
    }

    /**
     * 79. 单词搜索
     *
     * @param board
     * @param word
     * @return
     */
    public boolean exist(char[][] board, String word) {
        char[] charArray = word.toCharArray();
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[i].length; j++) {
                if (board[i][j] == charArray[0]) {
                    if (dfsExist(i, j, 0, board, charArray, new int[board.length][board[0].length])) return true;
                }
            }
        }
        return false;
    }

    private boolean dfsExist(int x, int y, int count, char[][] board, char[] charArray, int[][] visited) {
        if (visited[x][y] == 0 && board[x][y] == charArray[count]) {
            visited[x][y] = 1;
            count++;
            if (count == charArray.length) return true;
            if (x + 1 < board.length) {
                if (dfsExist(x + 1, y, count, board, charArray, visited)) return true;
            }
            if (y + 1 < board[0].length) {
                if (dfsExist(x, y + 1, count, board, charArray, visited)) return true;
            }
            if (x - 1 >= 0) {
                if (dfsExist(x - 1, y, count, board, charArray, visited)) return true;
            }
            if (y - 1 >= 0) {
                if (dfsExist(x, y - 1, count, board, charArray, visited)) return true;
            }
            visited[x][y] = 0;
        }
        return false;
    }

    /**
     * 437. 路径总和 III
     *
     * @param root
     * @param targetSum
     * @return
     */
    public int pathSum(TreeNode root, int targetSum) {
        if (root == null) {
            return 0;
        }

        int ret = rootSum(root, (long) targetSum);
        ret += pathSum(root.left, targetSum);
        ret += pathSum(root.right, targetSum);
        return ret;
    }

    public int rootSum(TreeNode root, Long targetSum) {
        int ret = 0;

        if (root == null) {
            return 0;
        }
        int val = root.val;
        if (val == targetSum) {
            ret++;
        }

        ret += rootSum(root.left, targetSum - val);
        ret += rootSum(root.right, targetSum - val);
        return ret;
    }

    /**
     * 347. 前 K 个高频元素
     *
     * @param nums
     * @param k
     * @return
     */
    public int[] topKFrequent(int[] nums, int k) {
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            map.put(nums[i], map.getOrDefault(nums[i], 0) + 1);
        }
        List<Integer>[] lists = new List[nums.length + 1];
        map.forEach((key, value) -> {
            if (lists[value] == null) {
                lists[value] = new ArrayList<>();
            }
            lists[value].add(key);
        });
        int count = 0;
        int[] res = new int[k];
        for (int i = lists.length - 1; i >= 0; i--) {
            if (lists[i] == null) continue;

            for (Integer integer : lists[i]) {
                res[count++] = integer;
                if (count == k) return res;
            }
        }
        return res;
    }


    /**
     * 236. 二叉树的最近公共祖先
     */
    private final HashMap<TreeNode, List<TreeNode>> aimAndParents = new HashMap<>();

    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null) return null;
        findNode(root, p);
        findNode(root, q);
        List<TreeNode> plist = aimAndParents.get(p);
        List<TreeNode> qlist = aimAndParents.get(q);
        if (plist == null) plist = new ArrayList<>();
        if (qlist == null) plist = new ArrayList<>();
        HashSet<TreeNode> set = new HashSet<>();
        for (TreeNode node : plist) {
            set.add(node);
        }
        for (TreeNode node : qlist) {
            if (set.contains(node)) return node;
        }
        return null;
    }

    public TreeNode findNode(TreeNode root, TreeNode aim) {
        if (root == null) return null;
        if (root == aim) {
            ArrayList<TreeNode> nodes = new ArrayList<>();
            nodes.add(root);
            aimAndParents.put(aim, nodes);
            return root;
        }
        TreeNode leftRes = findNode(root.left, aim);
        if (leftRes != null) {
            List<TreeNode> list = aimAndParents.get(aim);
            if (list == null) {
                list = new ArrayList<>();
                aimAndParents.put(aim, list);
            }
            list.add(root);
            return leftRes;
        }
        TreeNode rightRes = findNode(root.right, aim);
        if (rightRes != null) {
            List<TreeNode> list = aimAndParents.get(aim);
            if (list == null) {
                list = new ArrayList<>();
                aimAndParents.put(aim, list);
            }
            list.add(root);
            return rightRes;
        }
        return null;
    }

    /**
     * 287. 寻找重复数(可以使用环形链表来解决)
     *
     * @param nums
     * @return
     */
    public int findDuplicate(int[] nums) {
        int slow = 0;
        int fast = 0;
        while (true) {
            slow = nums[slow];
            fast = nums[nums[fast]];
            if (slow == fast) break;
        }
        fast = 0;
        while (true) {
            slow = nums[slow];
            fast = nums[fast];
            if (slow == fast) break;
        }
        return fast;
    }

    /**
     * 461. 汉明距离
     *
     * @param x
     * @param y
     * @return
     */
    public int hammingDistance(int x, int y) {
        return Integer.bitCount(x ^ y);
    }

    /**
     * 394. 字符串解码
     *
     * @param s
     * @return
     */
    public String decodeString(String s) {
        s = s + ']';
        return decodekuohaoString(s.toCharArray());
    }

    private int decodeStringIndex;

    public String decodekuohaoString(char[] chars) {
        StringBuilder stringBuilder = new StringBuilder();
        while (true) {
            if (Character.isDigit(chars[decodeStringIndex])) {
                int k = 0;
                while (Character.isDigit(chars[decodeStringIndex])) {
                    k = k * 10 + Integer.parseInt(String.valueOf(chars[decodeStringIndex++]));
                }
                decodeStringIndex += 1;
                String s = decodekuohaoString(chars);
                for (int j = 0; j < k; j++) {
                    stringBuilder.append(s);
                }
            } else if (Character.isLetter(chars[decodeStringIndex])) {
                stringBuilder.append(chars[decodeStringIndex]);
                decodeStringIndex++;
            } else if (chars[decodeStringIndex] == ']') {
                decodeStringIndex++;
                break;
            }
        }
        return stringBuilder.toString();
    }

    /**
     * 215. 数组中的第K个最大元素
     *
     * @param nums
     * @param k
     * @return
     */
    public int findKthLargest(int[] nums, int k) {
        return quickSortfindKthLargest(nums, 0, nums.length - 1, k);
    }

    public int quickSortfindKthLargest(int[] nums, int start, int end, int k) {
        if (start >= end) return nums[k];
        int l = start - 1;
        int r = end + 1;
        int p = nums[l + r >> 1];
        while (l < r) {
            do {
                l++;
            } while (nums[l] < p);
            do {
                r--;
            } while (nums[r] > p);
            if (l < r) {
                int temp = nums[l];
                nums[l] = nums[r];
                nums[r] = temp;
            }
        }
        if (k <= r) return quickSortfindKthLargest(nums, start, r, k);
        else return quickSortfindKthLargest(nums, r + 1, end, k);
    }

    /**
     * 240. 搜索二维矩阵 II
     *
     * @param matrix
     * @param target
     * @return
     */
    public boolean searchMatrix(int[][] matrix, int target) {
        int x = 0;
        int y = matrix[0].length - 1;
        while (x < matrix.length && y >= 0) {
            if (matrix[x][y] == target) return true;
            else if (matrix[x][y] > target) y--;
            else x++;
        }
        return false;
    }

    public int erfen(int[] nums, int aim) {
        int l = 0;
        int r = nums.length - 1;
        while (l < r) {
            int mid = l + r + 1 >> 1;
            if (nums[mid] <= aim) {
                l = mid;
            } else {
                r = mid - 1;
            }
        }
        return l;
    }


    /**
     * 207. 课程表(使用拓扑排序)
     *
     * @param numCourses
     * @param prerequisites
     * @return
     */
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        int[] rudu = new int[numCourses];
        HashMap<Integer, List<Integer>> map = new HashMap<>();
        for (int[] prerequisite : prerequisites) {
            if (!map.containsKey(prerequisite[1])) map.put(prerequisite[1], new ArrayList<>());
            map.get(prerequisite[1]).add(prerequisite[0]);//构建图之间的关系
            rudu[prerequisite[0]]++;//入度加一
        }
        Deque<Integer> deque = new ArrayDeque();
        for (int i = 0; i < rudu.length; i++) {
            //将所有入度为0的结点入栈
            if (rudu[i] == 0) {
                deque.offerLast(i);
            }
        }
        int count = 0;
        while (!deque.isEmpty()) {
            Integer last = deque.pollLast();
            //将该结点从图中去除,并将各个节点入度减一，如果入度被减为0就入栈
            count++;
            List<Integer> list = map.get(last);
            if (list == null) continue;
            for (Integer item : list) {
                rudu[item]--;
                if (rudu[item] == 0) {
                    deque.offerLast(item);
                }
            }
        }
        return count >= numCourses;
    }

    /**
     * 338. 比特位计数
     *
     * @param n
     * @return
     */
    public int[] countBits(int n) {
        if (n == 0) return new int[0];
        if (n == 1) return new int[]{0, 1};
        int[] res = new int[n + 1];
        res[0] = 0;
        res[1] = 1;
        for (int i = 2; i < res.length; i++) {
            if (i % 2 == 0) {
                res[i] = res[i >> 1];
            } else {
                res[i] = res[i - 1] + 1;
            }
        }
        return res;
    }

    /**
     * 146. LRU 缓存
     */
    class LRUCache {

        class Node {
            Integer val;
            Integer key;
            Node next;
            Node pre;

            public Node(Integer key, Integer val) {
                this.val = val;
                this.key = key;
            }
//            void show() {
//                Node temp = this;
//                while (temp != null) {
//
//                    System.out.print(" key: " + temp.key + " ");
//                    System.out.print("val: " + temp.val + " ->");
//
//                    temp = temp.next;
//                }
//                System.out.println();
//            }
        }

        HashMap<Integer, Node> map = new HashMap<>();
        int capacity;
        int count = 0;
        Node head;
        Node tail;

        public LRUCache(int capacity) {
            this.capacity = capacity;
            head = new Node(null, null);
            tail = new Node(null, null);
            head.next = tail;
            tail.pre = head;
        }

        public int get(int key) {
            Node node = map.get(key);
            if (node == null) {
                return -1;
            } else {
                removeNode(node);
                addToHead(node);
                return node.val;
            }
        }

        public void put(int key, int value) {
            if (map.containsKey(key)) {
                Node node = map.get(key);
                node.val = value;
                removeNode(node);
                addToHead(node);
            } else {
                if (count >= capacity) {
                    Node delelteNode = tail.pre;
                    removeNode(delelteNode);
                    map.remove(delelteNode.key);
                } else {
                    count++;
                }
                Node addNode = new Node(key, value);
                addToHead(addNode);
                map.put(key, addNode);
            }
        }

        void addToHead(Node node) {
            node.pre = head;
            node.next = head.next;
            head.next = node;
            node.next.pre = node;
        }

        void removeNode(Node node) {
            Node pre = node.pre;
            Node next = node.next;
            next.pre = pre;
            pre.next = next;
        }
    }

    /**
     * 647. 回文子串
     *
     * @param s
     * @return
     */
    public int countSubstrings(String s) {
        int count = 0;
        char[] charArray = s.toCharArray();
        for (int i = 0; i < charArray.length; i++) {
            for (int j = 0; j <= 1; j++) {
                int l = i;
                int r = i + j;
                while (l >= 0 && r < charArray.length && charArray[l] == charArray[r]) {
                    count++;
                    l--;
                    r++;
                }
            }
        }
        return count;
    }

    /**
     * 5. 最长回文子串
     *
     * @param s
     * @return
     */
    public String longestPalindrome(String s) {
        int maxLen = -1;
        String res = "";
        char[] charArray = s.toCharArray();
        for (int i = 0; i < charArray.length; i++) {
            for (int j = 0; j <= 1; j++) {
                int l = i;
                int r = i + j;
                while (l >= 0 && r < charArray.length && charArray[l] == charArray[r]) {
                    l--;
                    r++;
                }
                l++;
                r--;
                if (r - l > maxLen) {
                    maxLen = r - l;
                    res = s.substring(l, r + 1);
                }
            }
        }
        return res;
    }

    /**
     * 152. 乘积最大子数组
     *
     * @param nums
     * @return
     */
    public int maxProduct(int[] nums) {
        int[] minDp = new int[nums.length];
        int[] maxDp = new int[nums.length];
        maxDp[0] = nums[0];
        minDp[0] = nums[0];
        int max = nums[0];
        for (int i = 1; i < nums.length; i++) {
            //先处理maxDp
            int currentMax = Math.max(Math.max(nums[i], nums[i] * nums[i - 1]), Math.max(nums[i] * maxDp[i - 1], nums[i] * minDp[i - 1]));
            int currentMin = Math.min(Math.min(nums[i], nums[i] * nums[i - 1]), Math.min(nums[i] * maxDp[i - 1], nums[i] * minDp[i - 1]));
            minDp[i] = currentMin;
            maxDp[i] = currentMax;
            max = Math.max(currentMax, max);
        }
        return max;
    }

    /**
     * 238. 除自身以外数组的乘积
     */
    public int[] productExceptSelf(int[] nums) {
        if (nums.length <= 1) return nums;
        int[] incre = new int[nums.length];
        int[] decre = new int[nums.length];
        for (int i = 0; i < nums.length; i++) {
            if (i == 0) incre[i] = nums[i];
            else incre[i] = incre[i - 1] * nums[i];
        }
        for (int i = nums.length - 1; i >= 0; i--) {
            if (i == nums.length - 1) decre[i] = nums[i];
            else decre[i] = decre[i + 1] * nums[i];
        }
        int[] res = new int[nums.length];
        for (int i = 0; i < res.length; i++) {
            if (i == 0) res[i] = decre[i + 1];
            else if (i == res.length - 1) res[i] = incre[i - 1];
            else res[i] = incre[i - 1] * decre[i + 1];
        }
        return res;
    }

    /**
     * 221. 最大正方形
     *
     * @param matrix
     * @return
     */
    public int maximalSquare(char[][] matrix) {
//        int maxS=  0;
//        for (int i = 0; i < matrix.length; i++) {
//            for (int j = 0; j < matrix[i].length; j++) {
//                if (matrix[i][j] == '1') {
//                    for (int k = 0; true; k++) {
//                        int x = i + k;
//                        int xj = j;
//                        int y = j + k;
//                        int yi = i;
//                        if (x >= matrix.length || y >= matrix[0].length) break;
//                        while (matrix[x][xj] == '1' && matrix[yi][y] == '1') {
//                            if (x==yi&&xj==y){
//                                break;
//                            }
//                            yi++;xj++;
//                        }
//                        if (x==yi&&xj==y&&matrix[x][xj] == '1' && matrix[yi][y] == '1'){
//                            if (k>0){
//                                System.out.println("k = " + k);
//                                System.out.println("i = " + i);
//                                System.out.println("j = " + j);
//                                System.out.println( );
//                            }
//                            maxS = Math.max(maxS,(k+1)*(k+1));
//                        }else {
//                            break;
//                        }
//                    }
//                }
//            }
//        }
//        return maxS;
        int maxS = 0;
        int[][] dp = new int[matrix.length][matrix[0].length];
        for (int i = 0; i < dp[0].length; i++) {
            if (matrix[0][i] == '0') dp[0][i] = -1;
            else maxS = 1;
        }
        for (int i = 0; i < dp.length; i++) {
            if (matrix[i][0] == '0') dp[i][0] = -1;
            else maxS = 1;
        }
        for (int i = 1; i < matrix.length; i++) {
            for (int j = 1; j < matrix[i].length; j++) {
                int currentMaxLen = 0;
                while (i - currentMaxLen >= 0 && j - currentMaxLen >= 0 && matrix[i - currentMaxLen][j] == '1' && matrix[i][j - currentMaxLen] == '1') {
                    currentMaxLen++;
                }
                currentMaxLen--;
                currentMaxLen = Math.min(dp[i - 1][j - 1] + 1, currentMaxLen);
                dp[i][j] = currentMaxLen;
                currentMaxLen++;
                maxS = Math.max(currentMaxLen * currentMaxLen, maxS);
            }
        }
        return maxS;
    }


    /**
     * 621. 任务调度器
     */
    class LeastIntervalTask {
        int time;
        boolean state;
        Character val;
    }

    public int leastInterval(char[] tasks, int n) {
        int[] countArr = new int[26];
        for (char task : tasks) {
            countArr[task - 'A']++;
        }
        int maxCount = Integer.MIN_VALUE;
        int maxCountCOUNT = 0;
        for (int i = 0; i < countArr.length; i++) {
            if (countArr[i] > maxCount) {
                maxCount = countArr[i];
                maxCountCOUNT = 0;
            }
            if (countArr[i] == maxCount) maxCountCOUNT++;
        }
        return Math.max((maxCount - 1) * (n + 1) + maxCountCOUNT, tasks.length);
//        HashMap<Character,LeastIntervalTask> map = new HashMap<>();
//        int taskNum = 0;
//        for (int i = 0; i < tasks.length; i++) {
//            LeastIntervalTask leastIntervalTask = map.get(tasks[i]);
//            if (leastIntervalTask==null){
//                leastIntervalTask= new LeastIntervalTask();
//                leastIntervalTask.time = 0;
//                leastIntervalTask.val  = tasks[i];
//                map.put(tasks[i], leastIntervalTask);
//                taskNum++;
//            }
//            leastIntervalTask.time++;
//        }
//        List<LeastIntervalTask> list = new ArrayList<>();
//        for (LeastIntervalTask value : map.values()) {
//            list.add(value);
//        }
//        list.sort((o1, o2) -> Integer.compare(o2.time,o1.time));
//        Deque<LeastIntervalTask> waitDeque = new ArrayDeque<>();
//
//        int allTimes = 0;
//        while (list.size()>0){
//            LeastIntervalTask task = null;
//            int index = -1;
//            for (int i = 0; i < list.size(); i++) {
//               LeastIntervalTask currtask = list.get(i);
//               if (currtask.state==false){
//                   task = currtask;
//                   index = i;
//                   break;
//               }
//            }
//            if (task!=null){
//                task.state = true;
//                task.time--;
//                if (task.time<=0){
//                    list.remove(task);
//                }else {
//                    for (int i = 1; i+index < list.size(); i++) {
//                        LeastIntervalTask qian = list.get(index + i);
//                        LeastIntervalTask hou = list.get(index + i -1);
//                        if (hou.time< qian.time){
//                            list.set(index + i -1,qian);
//                            list.set(index+i,hou);
//                        }else {
//                            break;
//                        }
//                    }
//                }
//                    waitDeque.offerFirst(task);
//                    if (waitDeque.size()>n){
//                        LeastIntervalTask outTask = waitDeque.pollLast();
//                        outTask.state = false;
//                    }
//            }else {
//                waitDeque.offerFirst(new LeastIntervalTask());
//                if (waitDeque.size()>n){
//                    LeastIntervalTask outTask = waitDeque.pollLast();
//                    outTask.state = false;
//                }
//            }
//            allTimes++;
//        }
//        return allTimes;
    }

    /**
     * 448. 找到所有数组中消失的数字
     *
     * @param nums
     * @return
     */
    public List<Integer> findDisappearedNumbers(int[] nums) {
//        for (int i = 0; i < nums.length; i++) {
//            int nextIndex = nums[i]%nums.length;
//            int nextSValue = nums[i];
//            while (nums[nextIndex]%nums.length!=nextIndex){
//
//                int tempnextSValue = nums[nextIndex];
//                int tempnextIndex = nums[nextIndex]%nums.length;
//
//                nums[nextIndex] = nextSValue;
//
//                nextIndex = tempnextIndex;
//                nextSValue = tempnextSValue;
//
//            }
//        }
//        List<Integer> res = new ArrayList<>();
//        for (int i = 1; i <= nums.length; i++) {
//            if (nums[i%nums.length]%nums.length!=i%nums.length)res.add(i);
//        }
//        return res;

        int n = nums.length;
        for (int num : nums) {
            int x = (num - 1) % n;
            nums[x] += n;
        }
        List<Integer> ret = new ArrayList<Integer>();
        for (int i = 0; i < n; i++) {
            if (nums[i] <= n) {
                ret.add(i + 1);
            }
        }
        return ret;

    }


    /**
     * 297. 二叉树的序列化与反序列化
     */
    public static class Codec {

        // Encodes a tree to a single string.
        public String serialize(TreeNode root) {
            if (root == null) return "[]";
            Deque<TreeNode> deque = new LinkedList<>();
            deque.offerFirst(root);
            StringBuilder sb = new StringBuilder("[");
            while (!deque.isEmpty()) {
                TreeNode treeNode = deque.pollLast();
                sb.append(treeNode == null ? "null" : treeNode.val);
                sb.append(',');
                if (treeNode != null) {
                    deque.offerFirst(treeNode.left);
                    deque.offerFirst(treeNode.right);
                }
            }
            sb.replace(sb.length() - 1, sb.length(), "]");
            return sb.toString();
        }

        // Decodes your encoded data to tree.
        public TreeNode deserialize(String data) {
            if ("[]".equals(data)) return null;
            List<TreeNode> list = new ArrayList<>();
            char[] charArray = data.toCharArray();

            for (int i = 1; i < charArray.length; i++) {
                if (charArray[i] == 'n') {
                    i += 4;
                    list.add(null);
                } else {
                    TreeNode currentNode = new TreeNode();
                    if (charArray[i] == '-') {
                        i++;
                        while (charArray[i] != ',' && charArray[i] != ']') {
                            currentNode.val = currentNode.val * 10 - Integer.parseInt(String.valueOf(charArray[i]));
                            i++;
                        }
                    } else {
                        while (charArray[i] != ',' && charArray[i] != ']') {
                            currentNode.val = currentNode.val * 10 + Integer.parseInt(String.valueOf(charArray[i]));
                            i++;
                        }
                    }
                    list.add(currentNode);
                }
            }

            TreeNode root = list.get(0);
            if (root != null) {
                Deque<TreeNode> deque = new LinkedList<>();
                deque.offerFirst(root);
                int i = 1;
                while (!deque.isEmpty()) {
                    TreeNode node = deque.pollLast();
                    node.left = list.get(i++);
                    node.right = list.get(i++);
                    if (node.left != null) deque.offerFirst(node.left);
                    if (node.right != null) deque.offerFirst(node.right);
                }
            }
            return root;
        }
    }

    /**
     * 312. 戳气球
     *
     * @param nums
     * @return
     */
    public int maxCoins(int[] nums) {
        int[] balloons = new int[nums.length + 2];
        System.arraycopy(nums, 0, balloons, 1, nums.length);
        balloons[0] = 1;
        balloons[nums.length + 1] = 1;
        int[][] dp = new int[nums.length + 2][nums.length + 2];
        for (int len = 1; len <= nums.length; len++) {
            for (int l = 0; l + len + 1 <= nums.length + 1; l++) {
                int r = l + len + 1;
                for (int k = l + 1; k < r; k++) {
                    dp[l][r] = Math.max(dp[l][r], dp[l][k] + dp[k][r] + balloons[l] * balloons[k] * balloons[r]);
                }
            }
        }
        return dp[0][nums.length + 1];
    }

    /**
     * 438. 找到字符串中所有字母异位词
     *
     * @param s
     * @param p
     * @return
     */
    public List<Integer> findAnagrams(String s, String p) {
        long start = System.currentTimeMillis();
        if (s.length() < p.length()) return new ArrayList<>();
        char[] pCharArray = p.toCharArray();
        char[] sCharArray = s.toCharArray();
        int[] pvisited = new int[26];
        int[] svisited = new int[26];
        for (int i = 0; i < pCharArray.length; i++) {
            pvisited[pCharArray[i] - 'a']++;
        }
        for (int i = 0; i < pCharArray.length - 1; i++) {
            svisited[sCharArray[i] - 'a']++;
        }
        List<Integer> res = new ArrayList<>();
        for (int i = pCharArray.length - 1; i < sCharArray.length; i++) {
            svisited[sCharArray[i] - 'a']++;
            if (Arrays.equals(pvisited, svisited)) {
                res.add(i - pCharArray.length + 1);
            }
            svisited[sCharArray[i - pCharArray.length + 1] - 'a']--;
        }
        System.out.println("System.currentTimeMillis()-start = " + (System.currentTimeMillis() - start));
        return res;
    }

    /**
     * 把二叉搜索树转换为累加树
     *
     * @param root
     * @return
     */
    public TreeNode convertBST(TreeNode root) {
        int addNum = 0;
        Deque<TreeNode> deque = new ArrayDeque<>();
        TreeNode cur = root;
        while (cur != null || !deque.isEmpty()) {
            while (cur != null) {
                deque.offerLast(cur);
                cur = cur.right;
            }
            cur = deque.pollLast();
            cur.val += addNum;
            addNum = cur.val;
            cur = cur.left;
        }
        return root;
    }

    /**
     * 560. 和为 K 的子数组
     * @param nums
     * @param k
     * @return
     */
    public int subarraySum(int[] nums, int k) {
        Map<Integer,  Integer>  map = new HashMap<>();
        int[] qianzhuihe = new int[nums.length + 1];
        int count = 0;
        map.put(0,1);
        for (int i = 0; i < nums.length; i++) {
            qianzhuihe[i + 1] = qianzhuihe[i] + nums[i];
            count+=map.getOrDefault(qianzhuihe[i+1]-k,0);
            map.put(qianzhuihe[i+1],map.getOrDefault(qianzhuihe[i+1],0)+1);
        }
        return count;
    }

    /**
     * 33. 搜索旋转排序数组
     * @param nums
     * @param target
     * @return
     */
    public int search(int[] nums, int target) {
            int l = 0;
            int r = nums.length - 1;
            while (l<r){
                int mid = l+r>>1;
                if (nums[mid]==target){
                    return mid;
                }
                if(nums[mid]>target){
                    if (nums[mid]<nums[0]){
                        r = mid - 1;
                    }else {
                        if (target>=nums[0]){
                            r = mid - 1;
                        }else {
                            l = mid + 1;
                        }
                    }
                }else {
                     if (nums[mid]>=nums[0]){
                         l = mid+1;
                     }else {
                         if (target<nums[0]){
                             l = mid + 1;
                         }else {
                             r = mid - 1;
                         }
                     }
                }
            }
            return nums[l]==target ? l : -1;
    }

    public static void main(String[] args) {
        System.out.println("new Solution().search(new int[]{4,5,6,7,0,1,2},0) = " +
                new Solution().search(new int[]{4, 5, 6, 7, 0, 1, 2}, 0));
    }

}


















