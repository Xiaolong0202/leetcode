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

    /**
     * 454. 四数相加 II
     *
     * @param nums1
     * @param nums2
     * @param nums3
     * @param nums4
     * @return
     */
    public int fourSumCount(int[] nums1, int[] nums2, int[] nums3, int[] nums4) {
        Map<Integer, Integer> map = new HashMap();
        for (int i = 0; i < nums1.length; i++) {
            for (int j = 0; j < nums2.length; j++) {
                map.put(nums1[i] + nums2[j], map.getOrDefault(nums1[i] + nums2[j], 0) + 1);
            }
        }
        int res = 0;
        for (int i = 0; i < nums3.length; i++) {
            for (int j = 0; j < nums4.length; j++) {
                Integer num = map.get(-(nums3[i] + nums4[j]));
                if (num != null) {
                    res += num;
                }
            }
        }
        return res;
    }

    /**
     * 15. 三数之和
     *
     * @param nums
     * @return
     */
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        Arrays.sort(nums);
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] > 0) break;
            if (i > 0 && nums[i] == nums[i - 1]) continue;
            int aim = -nums[i];
            int l = i + 1;
            int r = nums.length - 1;
            while (l < r) {
                if (nums[l] + nums[r] == aim) {
                    res.add(Arrays.asList(nums[i], nums[l], nums[r]));
                    while (l < r && nums[l + 1] == nums[l]) {
                        l++;
                    }
                    while (l < r && nums[r - 1] == nums[r]) {
                        r--;
                    }
                    l++;
                    r--;
                } else if (nums[l] + nums[r] > aim) {
                    r--;
                } else if (nums[l] + nums[r] < aim) {
                    l++;
                }
            }
        }
        return res;
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

//    public String reverseWords(String s) {
//        s = s.trim();
//        StringBuilder sb = new StringBuilder();
//        char[] chars = s.toCharArray();
//        int len = chars.length;
//        int i = len - 1;
//        int j = len;
//        while (i >= 0) {
//            if (chars[i] == ' ') {
//                if ((i != len - 1) && chars[i + 1] == ' ') {
//                } else {
//                    sb.append(s, i + 1, j);
//                    sb.append(' ');
//                }
//                j = i;
//            } else if (i == 0) {
//                sb.append(s, i, j);
//            }
//            i--;
//        }
//        return sb.toString();
//    }


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

    /**
     * 738. 单调递增的数字
     *
     * @param n
     * @return
     */
    public int monotoneIncreasingDigits(int n) {
        char[] charArray = Integer.toString(n).toCharArray();
        for (int i = 0; i < charArray.length - 1; i++) {
            if (charArray[i] > charArray[i + 1]) {
                char c = charArray[i];
                i++;
                int j = i + 1;
                while (j < charArray.length) {
                    charArray[j++] = '9';
                }
                charArray[i--] = '9';
                while (i >= 0 && charArray[i] >= c) {
                    charArray[i--] = '9';
                }
                i++;
                charArray[i] = (char) (c - 1);
            }
        }
        return Integer.parseInt(new String(charArray));


//        String string = Integer.toString(n);
//        char[] num = string.toCharArray();
//        for (int i = 0; i < num.length - 1; i++) {
//            if (num[i] > num[i + 1]) {
//                while (i > 0 && num[i] == num[i - 1]) {
//                    i--;
//                }
//                num[i++]++;
//                while (i < num.length) {
//                    num[i++] = '9';
//                }
//                StringBuilder stringBuilder = new StringBuilder();
//                for (char c : num) {
//                    stringBuilder.append(c);
//                }
//                return Integer.parseInt(stringBuilder.toString());
//            }
//        }
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
        Arrays.fill(dp, Integer.MAX_VALUE - 10);
        dp[0] = 0;
        for (int i = 0; i < coins.length; i++) {
            for (int j = coins[i]; j < dp.length; j++) {
                dp[i] = Math.min(dp[j], dp[j - coins[i]] + 1);
            }
        }
        if (dp[amount] == Integer.MAX_VALUE - 10) {
            return -1;
        }
        return dp[amount];
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
    public boolean wordBreak1(String s, List<String> wordDict) {
        HashSet<String> set = new HashSet<>(wordDict);
        int length = s.length();
        boolean[] dp = new boolean[length + 1];//表示i之前的字符串是否能够被使用
        dp[0] = true;
        for (int i = 0; i < length; i++) {
            for (int j = i + 1; j <= length; j++) {
                if (dp[i] && set.contains(s.substring(i, j))) {
                    dp[j] = true;
                    if (j == length) {
                        return true;
                    }
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

//    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
//        if (root == null) return null;
//        findNode(root, p);
//        findNode(root, q);
//        List<TreeNode> plist = aimAndParents.get(p);
//        List<TreeNode> qlist = aimAndParents.get(q);
//        if (plist == null) plist = new ArrayList<>();
//        if (qlist == null) plist = new ArrayList<>();
//        HashSet<TreeNode> set = new HashSet<>();
//        for (TreeNode node : plist) {
//            set.add(node);
//        }
//        for (TreeNode node : qlist) {
//            if (set.contains(node)) return node;
//        }
//        return null;
//    }

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
        return findKthLargest(nums, 0, nums.length - 1, k);
    }


    public int findKthLargest(int[] nums, int l, int r, int k) {
        if (l == r) return nums[k];
        int p = nums[l + r >> 1];
        int i = l - 1;
        int j = r + 1;
        while (i < j) {
            do i++; while (p > nums[i]);
            do j--; while (p < nums[j]);
            if (i < j) {
                int temp = nums[i];
                nums[i] = nums[j];
                nums[j] = temp;
            }
        }
        if (k <= j) return findKthLargest(nums, l, j, k);
        else return findKthLargest(nums, j + 1, r, k);
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
     *
     * @param nums
     * @param k
     * @return
     */
    public int subarraySum(int[] nums, int k) {
        Map<Integer, Integer> map = new HashMap<>();
        int[] qianzhuihe = new int[nums.length + 1];
        int count = 0;
        map.put(0, 1);
        for (int i = 0; i < nums.length; i++) {
            qianzhuihe[i + 1] = qianzhuihe[i] + nums[i];
            count += map.getOrDefault(qianzhuihe[i + 1] - k, 0);
            map.put(qianzhuihe[i + 1], map.getOrDefault(qianzhuihe[i + 1], 0) + 1);
        }
        return count;
    }

    /**
     * 33. 搜索旋转排序数组
     *
     * @param nums
     * @param target
     * @return
     */
    public int search(int[] nums, int target) {
        int l = 0;
        int r = nums.length - 1;
        while (l < r) {
            int mid = l + r >> 1;
            if (nums[mid] == target) {
                return mid;
            }
            if (nums[mid] > target) {
                if (nums[mid] < nums[0]) {
                    r = mid - 1;
                } else {
                    if (target >= nums[0]) {
                        r = mid - 1;
                    } else {
                        l = mid + 1;
                    }
                }
            } else {
                if (nums[mid] >= nums[0]) {
                    l = mid + 1;
                } else {
                    if (target < nums[0]) {
                        l = mid + 1;
                    } else {
                        r = mid - 1;
                    }
                }
            }
        }
        return nums[l] == target ? l : -1;
    }

    /**
     * 54. 螺旋矩阵
     *
     * @param matrix
     * @return
     */
    public List<Integer> spiralOrder(int[][] matrix) {
        List<Integer> list = new ArrayList<>();
        int[][] visited = new int[matrix.length][matrix[0].length];
        int x = 0;
        int y = 0;

        while (true) {
            while (y < matrix[0].length && visited[x][y] == 0) {
                visited[x][y] = 1;
                list.add(matrix[x][y]);
                y++;
            }
            y--;
            x++;
            if (x < 0 || y < 0 || x >= matrix.length || y >= matrix[0].length) break;
            if (visited[x][y] == 1) break;
            while (x < matrix.length && visited[x][y] == 0) {
                visited[x][y] = 1;
                list.add(matrix[x][y]);
                x++;
            }
            x--;
            y--;
            if (x < 0 || y < 0 || x >= matrix.length || y >= matrix[0].length) break;
            if (visited[x][y] == 1) break;
            while (y >= 0 && visited[x][y] == 0) {
                visited[x][y] = 1;
                list.add(matrix[x][y]);
                y--;
            }
            y++;
            x--;
            if (x < 0 || y < 0 || x >= matrix.length || y >= matrix[0].length) break;
            if (visited[x][y] == 1) break;
            while (x >= 0 && visited[x][y] == 0) {
                visited[x][y] = 1;
                list.add(matrix[x][y]);
                x--;
            }
            x++;
            y++;
            if (x < 0 || y < 0 || x >= matrix.length || y >= matrix[0].length) break;
            if (visited[x][y] == 1) break;
        }
        return list;
    }

    /**
     * 516. 最长回文子序列
     *
     * @param s
     * @return
     */
    public int longestPalindromeSubseq(String s) {
        char[] charArray = s.toCharArray();
        int[][] dp = new int[charArray.length][charArray.length];//i~j之间是否存在回文序列
        for (int i = charArray.length - 1; i >= 0; i--) {
            for (int j = i; j < charArray.length; j++) {
                if (charArray[i] == charArray[j]) {
                    if (i == j) {
                        dp[i][j] = 1;
                    } else if (j - i == 1) {
                        dp[i][j] = 2;
                    } else {
                        dp[i][j] = dp[i + 1][j - 1] + 2;
                    }
                } else {
                    dp[i][j] = Math.max(dp[i + 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[0][dp.length - 1];
    }

    /**
     * 235. 二叉搜索树的最近公共祖先
     *
     * @param root
     * @param p
     * @param q
     * @return
     */
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        int small = Math.min(p.val, q.val);
        int big = Math.max(p.val, q.val);
        if (root.val >= small && root.val <= big) return root;
        else {
            if (root.val > big) {
                return lowestCommonAncestor(root.left, p, q);
            } else {
                return lowestCommonAncestor(root.right, p, q);
            }
        }
    }

    /**
     * 450. 删除二叉搜索树中的节点
     *
     * @param root
     * @param key
     * @return
     */
    public TreeNode deleteNode(TreeNode root, int key) {
        TreeNode keyNode = root;
        TreeNode pre = new TreeNode();
        pre.left = root;
        TreeNode res = pre;
        while (keyNode != null && keyNode.val != key) {
            pre = keyNode;
            if (keyNode.val > key) {
                keyNode = keyNode.left;
                continue;
            }
            if (keyNode.val < key) {
                keyNode = keyNode.right;
                continue;
            }
        }
        if (keyNode == null) return root;
        TreeNode right = keyNode.right;
        TreeNode temp = right;
        if (right != null) {
            while (temp.left != null) {
                temp = temp.left;
            }
            temp.left = keyNode.left;
            if (pre.left == keyNode) {
                pre.left = right;
            } else {
                pre.right = right;
            }
        } else {
            if (pre.left == keyNode) {
                pre.left = keyNode.left;
            } else {
                pre.right = keyNode.left;
            }
        }
        return res.left;
    }

    /**
     * 72. 编辑距离
     *
     * @param word1
     * @param word2
     * @return
     */
    public int minDistance(String word1, String word2) {
        char[] word1CharArray = word1.toCharArray();
        char[] word2CharArray = word2.toCharArray();
        int dp[][] = new int[word1CharArray.length + 1][word2CharArray.length + 1];
        for (int i = 1; i < dp.length; i++) {
            dp[i][0] = i;
        }
        for (int i = 1; i < dp[0].length; i++) {
            dp[0][i] = i;
        }
        for (int i = 1; i < dp.length; i++) {
            for (int j = 1; j < dp[0].length; j++) {
                if (word1CharArray[i - 1] == word2CharArray[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = Math.min(Math.min(dp[i - 1][j], dp[i][j - 1]), dp[i - 1][j - 1]) + 1;
                }
            }
        }
        return dp[word1.length()][word2.length()];
    }

    public void guibingSort(int[] nums, int l, int r) {
        if (l >= r) return;
        int mid = l + r >> 1;
        guibingSort(nums, l, mid);
        guibingSort(nums, mid + 1, r);
        int i = l;
        int j = mid + 1;
        int[] temp = new int[r - l + 1];
        int tempIndex = 0;
        while (i <= mid && j <= r) {
            if (nums[i] <= nums[j]) {
                temp[tempIndex++] = nums[i++];
            } else {
                temp[tempIndex++] = nums[j++];
            }
        }
        while (i <= mid) {
            temp[tempIndex++] = nums[i++];
        }
        while (j <= r) {
            temp[tempIndex++] = nums[j++];
        }
        tempIndex--;
        while (tempIndex >= 0) {
            nums[r--] = temp[tempIndex--];
        }
    }


    /**
     * 513. 找树左下角的值
     *
     * @param root
     * @return
     */
    private int maxN = -1;
    private Map<Integer, Integer> findBottomLeftValuemap = new HashMap<>();

    public int findBottomLeftValue(TreeNode root) {
        findBottomLeftValue(root, 0);
        return findBottomLeftValuemap.get(maxN);
    }

    public void findBottomLeftValue(TreeNode root, int n) {
        if (root == null) return;
        if (!findBottomLeftValuemap.containsKey(n)) {
            findBottomLeftValuemap.put(n, root.val);
        }
        maxN = Math.max(maxN, n);
        findBottomLeftValue(root.left, n + 1);
        findBottomLeftValue(root.right, n + 1);
    }

    /**
     * 112. 路径总和
     *
     * @param root
     * @param targetSum
     * @return
     */
    public boolean hasPathSum(TreeNode root, int targetSum) {
        if (root == null) return false;
        if (root.left == null && root.right == null && root.val == targetSum) return true;
        return hasPathSum(root.left, targetSum - root.val) || hasPathSum(root.right, targetSum - root.val);
    }

    /**
     * 654. 最大二叉树
     *
     * @param nums
     * @return
     */
    public TreeNode constructMaximumBinaryTree(int[] nums) {
        return constructMaximumBinaryTree(nums, 0, nums.length - 1);
    }

    public TreeNode constructMaximumBinaryTree(int[] nums, int start, int end) {
        TreeNode root = new TreeNode();
        if (start == end) {
            root.val = nums[start];
            return root;
        }
        int maxIndex = start;
        for (int i = start; i <= end; i++) {
            if (nums[i] > nums[maxIndex]) {
                maxIndex = i;
            }
        }
        root.val = nums[maxIndex];
        if (maxIndex > start) {
            root.left = constructMaximumBinaryTree(nums, start, maxIndex - 1);
        }
        if (maxIndex < end) {
            root.right = constructMaximumBinaryTree(nums, maxIndex + 1, end);
        }
        return root;
    }

    /**
     * 968. 监控二叉树
     *
     * @param root
     * @return
     */
    private int count = 0;

    public int minCameraCover(TreeNode root) {
        if (travlese(root) == 0) count++;
        return count;
    }

    //0 未覆盖  1 有摄像机 2已经覆盖 状态转移
    public int travlese(TreeNode root) {
        if (root == null) return 2;
        int left = travlese(root.left);
        int right = travlese(root.right);
        if (left == 0 || right == 0) {
            count++;
            return 1;
        }
        if (left == 1 || right == 1) return 2;
        return 0;
    }

    /**
     * 12. 整数转罗马数字
     *
     * @return
     */
    class LUOMA {
        Integer num;
        String str;

        public LUOMA(Integer num, String str) {
            this.num = num;
            this.str = str;
        }
    }

    /**
     * 整数转罗马数字
     *
     * @param num
     * @return
     */
    public String intToRoman(int num) {
        StringBuilder stringBuilder = new StringBuilder();
        List<LUOMA> list = new ArrayList<>();
        list.add(new LUOMA(1000, "M"));
        list.add(new LUOMA(900, "CM"));
        list.add(new LUOMA(500, "D"));
        list.add(new LUOMA(400, "CD"));
        list.add(new LUOMA(100, "C"));
        list.add(new LUOMA(90, "XC"));
        list.add(new LUOMA(50, "L"));
        list.add(new LUOMA(40, "XL"));
        list.add(new LUOMA(10, "X"));
        list.add(new LUOMA(9, "IX"));
        list.add(new LUOMA(5, "V"));
        list.add(new LUOMA(4, "IV"));
        list.add(new LUOMA(1, "I"));
        for (LUOMA luoma : list) {
            while (luoma.num <= num) {
                num -= luoma.num;
                stringBuilder.append(luoma.str);
            }
        }
        return stringBuilder.toString();
    }

    /**
     * 24. 两两交换链表中的节点
     *
     * @param head
     * @return
     */
    public ListNode swapPairs(ListNode head) {
        if (head == null || head.next == null) return head;
        ListNode p = new ListNode();//结点p表示前驱结点，即 p.next与p.next.next 进行交换操作，两个结点交换完成之后再接到p结点之后
        p.next = head;
        head = p;

        while (p.next != null && p.next.next != null) {
            ListNode qian = p.next;
            ListNode hou = qian.next;

            qian.next = hou.next;
            hou.next = qian;
            p.next = hou;

            p = qian;
        }
        return head.next;
    }


    /**
     * 150. 逆波兰表达式求值
     *
     * @param tokens
     * @return
     */
    public int evalRPN(String[] tokens) {
        Deque<String> deque = new ArrayDeque<>();
        for (int i = 0; i < tokens.length; i++) {
            if (Character.isDigit(tokens[i].charAt(0)) || tokens[i].length() > 1) {
                deque.offerLast(tokens[i]);
            } else {
                int b = Integer.parseInt(deque.pollLast());
                int a = Integer.parseInt(deque.pollLast());
                switch (tokens[i]) {
                    case "+":
                        deque.offerLast(String.valueOf(a + b));
                        break;
                    case "-":
                        deque.offerLast(String.valueOf(a - b));
                        break;
                    case "*":
                        deque.offerLast(String.valueOf(a * b));
                        break;
                    case "/":
                        deque.offerLast(String.valueOf(a / b));
                        break;
                }
            }
        }
        return Integer.parseInt(deque.pollLast());
    }

    /**
     * 1047. 删除字符串中的所有相邻重复项
     *
     * @param s
     * @return
     */
    public String removeDuplicates(String s) {
        Deque<Character> deque = new ArrayDeque<>();
        char[] chars = s.toCharArray();
        for (char c : chars) {
            if (!deque.isEmpty() && deque.peekLast() == c) {
                deque.pollLast();
            } else {
                deque.offerLast(c);
            }
        }
        StringBuilder sb = new StringBuilder();
        while (!deque.isEmpty()) {
            sb.append(deque.pollFirst());
        }
        return sb.toString();
    }


    /**
     * 93. 复原 IP 地址
     *
     * @param s
     * @return
     */
    public List<String> restoreIpAddresses(String s) {
        List<String> list = new ArrayList<>();
        restoreIpAddresses(s.toCharArray(), 4, 0, new StringBuilder(), list);
        return list;
    }

    public void restoreIpAddresses(char[] str, int n, int i, StringBuilder sb, List<String> res) {
        if (i == str.length) {
            if (n == 0) {
                sb.delete(sb.length() - 1, sb.length());
                res.add(sb.toString());
                sb.append(".");
            }
            return;
        }
        if (str.length - i > n * 3) return; //长度不匹配则放弃
        if (str[i] == '0') {
            sb.append(str[i]);
            sb.append('.');
            restoreIpAddresses(str, n - 1, i + 1, sb, res);
            sb.delete(sb.length() - 2, sb.length());
            return;
        }
        for (int j = 0; j < 3 && i + j < str.length; j++) {
            String appendValue = new String(str, i, j + 1);
            if (Integer.parseInt(appendValue) <= 255) {
                sb.append(appendValue);
                sb.append('.');
                restoreIpAddresses(str, n - 1, i + appendValue.length(), sb, res);
                sb.delete(sb.length() - appendValue.length() - 1, sb.length());
            }
        }
    }


    public void nextPermutation(int[] nums) {
        int i = nums.length - 1;
        while (i > 0 && nums[i] <= nums[i - 1]) {
            //todo  交换
            i--;
        }
        if (i == 0) {
            int l = i;
            int r = nums.length - 1;
            while (l < r) {
                int temp = nums[l];
                nums[l] = nums[r];
                nums[r] = temp;
                l++;
                r--;
            }
            return;
        }
        i--;
        int j = nums.length - 1;
        while (nums[j] <= nums[i]) {
            j--;
        }
        int temp = nums[j];
        nums[j] = nums[i];
        nums[i] = temp;

        i++;
        j = nums.length - 1;
        while (i < j) {
            temp = nums[j];
            nums[j] = nums[i];
            nums[i] = temp;
            i++;
            j--;
        }
    }

    /**
     * 2. 两数相加
     *
     * @param l1
     * @param l2
     * @return
     */
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        int extra = 0;
        ListNode pre = new ListNode();
        ListNode head = pre;
        while (l2 != null && l1 != null) {
            ListNode cur = new ListNode();
            cur.val = (l1.val + l2.val + extra) % 10;
            extra = (l1.val + l2.val + extra) / 10;
            pre.next = cur;
            pre = pre.next;
            l2 = l2.next;
            l1 = l1.next;
        }
        while (l1 != null) {
            ListNode cur = new ListNode();
            cur.val = (l1.val + extra) % 10;
            extra = (l1.val + extra) / 10;
            pre.next = cur;
            pre = pre.next;
            l1 = l1.next;
        }
        while (l2 != null) {
            ListNode cur = new ListNode();
            cur.val = (l2.val + extra) % 10;
            extra = (l2.val + extra) / 10;
            pre.next = cur;
            pre = pre.next;
            l2 = l2.next;
        }
        if (extra > 0) {
            ListNode cur = new ListNode();
            cur.val = extra;
            pre.next = cur;
        }
        return head.next;
    }

    /**
     * 41. 缺失的第一个正数
     *
     * @param nums
     * @return
     */
    public int firstMissingPositive(int[] nums) {
        for (int i = 0; i < nums.length; i++) {
            while (nums[i] > 0 && nums[i] <= nums.length && nums[i] != nums[nums[i] - 1]) {
                int temp = nums[i];
                nums[i] = nums[temp - 1];
                nums[temp - 1] = temp;
            }
        }


        for (int i = 0; i < nums.length; i++) {
            if (nums[i] != i + 1) return i + 1;
        }
        return nums.length + 1;
    }

    /**
     * @param head
     * @param x
     * @return
     */
    public ListNode partition(ListNode head, int x) {
        ListNode pre = new ListNode();
        pre.next = head;
        head = pre;
        while (pre.next != null && pre.next.val < x) {
            pre = pre.next;
        }
        if (pre.next == null) return head.next;
        //找到>=x结点的前驱结点
        ListNode lpre = pre.next;
        ListNode l = lpre.next;
        while (l != null) {
            if (l.val < x) {

                lpre.next = l.next;

                l.next = pre.next;
                pre.next = l;
                pre = pre.next;

                l = lpre.next;
            } else {
                l = l.next;
                lpre = lpre.next;
            }
        }
        return head.next;
    }

    /**
     * 165. 比较版本号
     *
     * @param version1
     * @param version2
     * @return
     */
    public int compareVersion(String version1, String version2) {
        String[] split1 = version1.split("\\.");
        String[] split2 = version2.split("\\.");
        int i = 0;
        int j = 0;
        while (i < split1.length && j < split2.length) {
            int i1 = Integer.parseInt(split1[i++]);
            int i2 = Integer.parseInt(split2[j++]);
            if (i1 > i2) return 1;
            if (i2 > i1) return -1;
        }
        while (i < split1.length) {
            int i1 = Integer.parseInt(split1[i++]);
            if (i1 > 0) return 1;
        }
        while (j < split2.length) {
            int i2 = Integer.parseInt(split2[j++]);
            if (i2 > 0) return -1;
        }
        return 0;
    }

    /**
     * 145. 二叉树的后序遍历
     *
     * @param root
     * @return
     */
    public List<Integer> postorderTraversal(TreeNode root) {
        List<Integer> list = new ArrayList<>();
        Deque<TreeNode> deque = new LinkedList<>();
        deque.offerLast(root);
        while (!deque.isEmpty()) {
            TreeNode last = deque.pollLast();
            if (last == null) {
                continue;
            }
            deque.offerLast(last.left);
            deque.offerLast(last.right);
            list.add(0, last.val);
        }
        return list;
    }

    /**
     * 61. 旋转链表
     *
     * @param head
     * @param k
     * @return
     */
    public ListNode rotateRight(ListNode head, int k) {
        if (head == null) return head;
        ListNode t = head;
        int len = 0;
        while (t != null) {
            len++;
            t = t.next;
        }

        ListNode slow = head;
        ListNode fast = head;
        for (int i = 0; i < k % len; i++) {
            if (fast.next == null) {
                fast = head;
            } else {
                fast = fast.next;
            }
        }
        while (fast.next != null) {
            fast = fast.next;
            slow = slow.next;
        }
        fast.next = head;
        head = slow.next;
        slow.next = null;

        return head;
    }

    /**
     * 752. 打开转盘锁
     */
    class openLockTime {
        String string;
        int times = 0;

        public openLockTime(String string, int times) {
            this.string = string;
            this.times = times;
        }
    }

    public int openLock(String[] deadends, String target) {
        HashSet<String> set = new HashSet<>();
        for (String deadend : deadends) {
            set.add(deadend);
        }

        Deque<openLockTime> deque = new ArrayDeque<>();
        HashSet<String> visited = new HashSet<>();

        deque.offerLast(new openLockTime("0000", 0));
        while (!deque.isEmpty()) {
            openLockTime openLockTime = deque.pollFirst();
            if (set.contains(openLockTime.string) || visited.contains(openLockTime.string)) {
                continue;
            }
            if (openLockTime.string.equals(target)) {
                return openLockTime.times;
            }
            visited.add(openLockTime.string);
            char[] charArray = openLockTime.string.toCharArray();
            for (int i = 0; i < charArray.length; i++) {
                //++
                if (charArray[i] == '9') {
                    charArray[i] = '0';
                    deque.offerLast(new openLockTime(new String(charArray), openLockTime.times + 1));
                    charArray[i] = '9';
                } else {
                    charArray[i]++;
                    deque.offerLast(new openLockTime(new String(charArray), openLockTime.times + 1));
                    charArray[i]--;
                }
                //--
                if (charArray[i] == '0') {
                    charArray[i] = '9';
                    deque.offerLast(new openLockTime(new String(charArray), openLockTime.times + 1));
                    charArray[i] = '0';
                } else {
                    charArray[i]--;
                    deque.offerLast(new openLockTime(new String(charArray), openLockTime.times + 1));
                    charArray[i]++;
                }
            }
        }
        return -1;
    }


    /**
     * 岛屿的最大面积
     *
     * @param grid
     * @return
     */
    public int maxAreaOfIsland(int[][] grid) {
        int maxS = 0;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[i].length; j++) {
                if (grid[i][j] == 1) {
                    int currentS = 0;
                    Deque<int[]> deque = new ArrayDeque<>();
                    deque.offerLast(new int[]{i, j});
                    while (!deque.isEmpty()) {
                        int[] last = deque.pollFirst();
                        int x = last[0];
                        int y = last[1];
                        if (grid[x][y] == 1) {
                            grid[x][y] = 0;
                            currentS++;
                            if (x - 1 >= 0) {
                                deque.offerLast(new int[]{x - 1, y});
                            }
                            if (y - 1 >= 0) {
                                deque.offerLast(new int[]{x, y - 1});
                            }
                            if (y + 1 < grid[0].length) {
                                deque.offerLast(new int[]{x, y + 1});
                            }
                            if (x + 1 < grid.length) {
                                deque.offerLast(new int[]{x + 1, y});
                            }
                        }
                    }
                    maxS = Math.max(maxS, currentS);
                }
            }
        }
        return maxS;
    }

    /**
     * 1004. 最大连续1的个数 III
     *
     * @param nums
     * @param k
     * @return
     */
    public int longestOnes(int[] nums, int k) {
        int l = 0;
        int r = 0;
        int currentKUse = 0;
        int res = 0;
        while (r < nums.length) {
            if (nums[r] == 0) {
                currentKUse++;
            }
            while (l <= r && currentKUse > k) {
                if (nums[l++] == 0) {
                    currentKUse--;
                }
            }
            r++;
            res = Math.max(res, r - l);
        }
        return res;
    }

    /**
     * 797. 所有可能的路径
     *
     * @param graph
     * @return
     */
    public List<List<Integer>> allPathsSourceTarget(int[][] graph) {
        List<List<Integer>> res = new ArrayList<>();
        ArrayList<Integer> list = new ArrayList<>();
        list.add(0);
        allPathsSourceTarget(graph, res, list, 0);
        return res;
    }

    public void allPathsSourceTarget(int[][] graph, List<List<Integer>> res, List<Integer> list, int currentIndex) {
        if (currentIndex == graph.length - 1) {
            res.add(new ArrayList<>(list));
            return;
        }
        for (int i = 0; i < graph[currentIndex].length; i++) {
            list.add(graph[currentIndex][i]);
            allPathsSourceTarget(graph, res, list, graph[currentIndex][i]);
            list.remove(list.size() - 1);
        }
    }

    public int numEnclaves(int[][] grid) {
        int sumS = 0;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[i].length; j++) {
                if (grid[i][j] == 1) {
                    Deque<int[]> deque = new ArrayDeque<>();
                    deque.offerLast(new int[]{i, j});
                    int s = 0;
                    boolean flag = false;
                    while (!deque.isEmpty()) {
                        int[] last = deque.pollLast();
                        int x = last[0];
                        int y = last[1];
                        if (grid[x][y] == 1) {
                            System.out.println("s = " + s);
                            System.out.println("y = " + y);
                            grid[x][y] = 0;
                            s++;
                            if (x == 0 || y == 0 || x == grid.length - 1 || y == grid[0].length - 1) {
                                flag = true;
                            }
                            if (x - 1 >= 0) {
                                deque.offerLast(new int[]{x - 1, y});
                            }
                            if (y - 1 >= 0) {
                                deque.offerLast(new int[]{x, y - 1});
                            }
                            if (y + 1 < grid[0].length) {
                                deque.offerLast(new int[]{x, y + 1});
                            }
                            if (x + 1 < grid.length) {
                                deque.offerLast(new int[]{x + 1, y});
                            }
                        }
                    }
                    if (!flag) {
                        sumS += s;
                    }
                }
            }
        }
        return sumS;
    }

    /**
     * 130. 被围绕的区域
     *
     * @param board
     */
    public void solve(char[][] board) {
        int[][] vivited = new int[board.length][board[0].length];
        for (int i = 0; i < board.length; i++) {
            solveDfs(board, vivited, i, 0);
            solveDfs(board, vivited, i, board[0].length - 1);
        }
        for (int i = 0; i < board[0].length; i++) {
            solveDfs(board, vivited, 0, i);
            solveDfs(board, vivited, board.length - 1, i);
        }
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                if (board[i][j] == 'O' && vivited[i][j] == 0) board[i][j] = 'X';
            }
        }
    }

    public void solveDfs(char[][] board, int[][] visited, int x, int y) {
        if (x < 0 || y < 0 || x >= board.length || y >= board[0].length || board[x][y] == 'X' || visited[x][y] == 1) {
            return;
        }
        visited[x][y] = 1;
        solveDfs(board, visited, x + 1, y);
        solveDfs(board, visited, x - 1, y);
        solveDfs(board, visited, x, y + 1);
        solveDfs(board, visited, x, y - 1);
    }

    /**
     * 417. 太平洋大西洋水流问题
     *
     * @param heights
     * @return
     */
    public List<List<Integer>> pacificAtlantic(int[][] heights) {
        boolean[][] taiping = new boolean[heights.length][heights[0].length];
        boolean[][] daxi = new boolean[heights.length][heights[0].length];
        for (int i = 0; i < heights.length; i++) {
            dfspacificAtlanticTAIPING(heights, taiping, i, 0, i, 0);
            dfspacificAtlanticTAIPING(heights, daxi, i, heights[0].length - 1, i, heights[0].length - 1);
        }
        for (int i = 0; i < heights[0].length; i++) {
            dfspacificAtlanticTAIPING(heights, taiping, 0, i, 0, i);
            dfspacificAtlanticTAIPING(heights, daxi, heights.length - 1, i, heights.length - 1, i);
        }
        List<List<Integer>> res = new ArrayList<>();
        for (int i = 0; i < heights.length; i++) {
            for (int j = 0; j < heights[0].length; j++) {
                if (taiping[i][j] && daxi[i][j]) {
                    res.add(List.of(i, j));
                }
            }
        }
        return res;
    }

    private void dfspacificAtlanticTAIPING(int[][] heights, boolean[][] taiping, int x, int y, int preX, int preY) {
        if (x < 0 || y < 0 || x >= heights.length || y >= heights[0].length) return;
        if (taiping[x][y]) return;
        if (heights[x][y] < heights[preX][preY]) {
            return;
        }
        taiping[x][y] = true;
        dfspacificAtlanticTAIPING(heights, taiping, x + 1, y, x, y);
        dfspacificAtlanticTAIPING(heights, taiping, x, y + 1, x, y);
        dfspacificAtlanticTAIPING(heights, taiping, x - 1, y, x, y);
        dfspacificAtlanticTAIPING(heights, taiping, x, y - 1, x, y);
    }

    /**
     * 127. 单词接龙
     *
     * @param beginWord
     * @param endWord
     * @param wordList
     * @return
     */
    public int ladderLength(String beginWord, String endWord, List<String> wordList) {
        HashSet<String> set = new HashSet<>(wordList);
        Deque<String> deque = new ArrayDeque<>();
        Map<String, Integer> map = new HashMap<>();
        map.put(beginWord, 1);
        deque.offerLast(beginWord);
        while (!deque.isEmpty()) {
            String last = deque.pollFirst();
            Integer path = map.get(last);
            if (last.equals(endWord)) {
                return path;
            }
            char[] charArray = last.toCharArray();
            for (int i = 0; i < charArray.length; i++) {
                for (int j = 'a'; j <= 'z'; j++) {
                    char c = charArray[i];
                    charArray[i] = (char) j;
                    String s = String.valueOf(charArray);
                    if (set.contains(s) && !map.containsKey(s)) {
                        map.put(s, path + 1);
                        deque.offerLast(s);
                    }
                    charArray[i] = c;
                }
            }
        }
        return 0;
    }

    /**
     * 841. 钥匙和房间
     *
     * @param rooms
     * @return
     */
    public boolean canVisitAllRooms(List<List<Integer>> rooms) {
        HashSet<Integer> keySet = new HashSet<>();
        Deque<Integer> deque = new ArrayDeque<>();
        deque.offerLast(0);
        while (!deque.isEmpty()) {
            Integer room = deque.pollLast();
            keySet.add(room);
            if (keySet.size() == rooms.size()) {
                return true;
            }
            for (Integer next : rooms.get(room)) {
                if (!keySet.contains(next)) {
                    deque.offerLast(next);
                }
            }
        }
        return false;
    }

    /**
     * 463. 岛屿的周长
     */
    private int islandPerimeterLong = 0;

    public int islandPerimeter(int[][] grid) {
        int x = -1;
        int y = -1;
        ok:
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == 1) {
                    x = i;
                    y = j;
                    break ok;
                }
            }
        }
        dfsIslandPerimeter(grid, x, y, new int[grid.length][grid[0].length]);
        return islandPerimeterLong;
    }

    boolean dfsIslandPerimeter(int[][] grid, int x, int y, int[][] visited) {
        if (x < 0 || y < 0 || x >= grid.length || y >= grid[0].length) {
            return true;
        }
        if (grid[x][y] == 0) {
            return true;
        }
        if (grid[x][y] == 1) {
            if (visited[x][y] == 1) {
                return false;
            }
            visited[x][y] = 1;
            if (dfsIslandPerimeter(grid, x + 1, y, visited)) {
                islandPerimeterLong++;
            }
            if (dfsIslandPerimeter(grid, x - 1, y, visited)) {
                islandPerimeterLong++;
            }
            if (dfsIslandPerimeter(grid, x, y + 1, visited)) {
                islandPerimeterLong++;
            }
            if (dfsIslandPerimeter(grid, x, y - 1, visited)) {
                islandPerimeterLong++;
            }
        }
        return false;
    }

    /**
     * 1971. 寻找图中是否存在路径
     *
     * @param n
     * @param edges
     * @param source
     * @param destination
     * @return
     */
    public boolean validPath(int n, int[][] edges, int source, int destination) {
        UnionFindAlgorithm unionFindAlgorithm = new UnionFindAlgorithm();
        for (int i = 0; i < edges.length; i++) {
            unionFindAlgorithm.join(edges[i][0], edges[i][1]);
        }
        return unionFindAlgorithm.find(source) == unionFindAlgorithm.find(destination);
    }


    /**
     * 684. 冗余连接
     *
     * @param edges
     * @return
     */
    public int[] findRedundantConnection(int[][] edges) {
        UnionFindAlgorithm unionFindAlgorithm = new UnionFindAlgorithm();
        for (int i = 0; i < edges.length; i++) {
            if (unionFindAlgorithm.find(edges[i][0]) == unionFindAlgorithm.find(edges[i][1])) {
                return new int[]{edges[i][0], edges[i][1]};
            }
            unionFindAlgorithm.join(edges[i][0], edges[i][1]);
        }
        return new int[2];
    }

    /**
     * 685. 冗余连接 II
     *
     * @param edges
     * @return
     */
    public int[] findRedundantDirectedConnection(int[][] edges) {
        UnionFindAlgorithm unionFindAlgorithm = new UnionFindAlgorithm();
        for (int i = 0; i < edges.length; i++) {
            if (unionFindAlgorithm.find(edges[i][0]) == unionFindAlgorithm.find(edges[i][1])) {
                return new int[]{edges[i][0], edges[i][1]};
            }
            unionFindAlgorithm.join(edges[i][0], edges[i][1]);
        }
        return new int[2];
    }


    /**
     * 16. 最接近的三数之和
     *
     * @param nums
     * @param target
     * @return
     */
    public int threeSumClosest(int[] nums, int target) {
        int minMinus = 0x3f3f3f3f;
        Arrays.sort(nums);
        for (int i = 0; i < nums.length - 2; i++) {
            if (i > 0 && nums[i] == nums[i - 1]) continue;
            int l = i + 1;
            int r = nums.length - 1;
            int aim = target - nums[i];
            while (l < r) {
                int tempSum = nums[l] + nums[r];
                if (Math.abs(minMinus) > Math.abs(aim - tempSum)) {
                    minMinus = aim - tempSum;
                }
                if (tempSum > aim) {
                    r--;
                }
                if (tempSum < aim) {
                    l++;
                }
                if (tempSum == aim) {
                    return target;
                }
            }
        }
        return target - minMinus;
    }

    /**
     * 399. 除法求值
     *
     * @param equations
     * @param values
     * @param queries
     * @return
     */
    public double[] calcEquation(List<List<String>> equations, double[] values, List<List<String>> queries) {
        Map<String, String> parent = new HashMap<>();//对应父级
        Map<String, Double> weight = new HashMap<>();//对应到父级的权重,比值

        //初始
        equations.forEach(list -> {
            parent.put(list.get(0), list.get(0));
            parent.put(list.get(1), list.get(1));

            weight.put(list.get(0), 1D);
            weight.put(list.get(1), 1D);
        });
        for (int i = 0; i < values.length; i++) {
            List<String> equationList = equations.get(i);
            String up = equationList.get(0);
            String down = equationList.get(1);
            add(up, down, values[i], parent, weight);
        }
        double[] res = new double[queries.size()];
        for (int i = 0; i < res.length; i++) {
            List<String> queiesList = queries.get(i);
            String up = queiesList.get(0);
            String down = queiesList.get(1);
            if (parent.get(up) == null || parent.get(down) == null || !find(up, parent, weight).equals(find(down, parent, weight))) {
                res[i] = -1D;
                continue;
            }
            res[i] = weight.get(up) / weight.get(down);
        }
        return res;
    }

    String find(String a, Map<String, String> parent, Map<String, Double> weight) {
        if (parent.containsKey(a) && !parent.get(a).equals(a)) {
            String p = parent.get(a);
            parent.put(a, find(p, parent, weight));
            weight.put(a, weight.get(p) * weight.get(a));
        }
        return parent.get(a);
    }

    //a -> b
    void add(String a, String b, Double v, Map<String, String> parent, Map<String, Double> weight) {
        String pA = find(a, parent, weight);
        String pB = find(b, parent, weight);
        if (pA.equals(pB)) {
            return;
        }
        weight.put(pA, v * weight.get(b) / weight.get(a));
        parent.put(pA, pB);
    }

    /**
     * 58. 最后一个单词的长度
     *
     * @param s
     * @return
     */
    public int lengthOfLastWord(String s) {
        char[] charArray = s.toCharArray();
        int l = -1;
        for (int i = 0; i < charArray.length; i++) {
            if (charArray[i] == ' ' && charArray[Math.min(i + 1, charArray.length - 1)] != ' ') {
                l = i;
            }
        }
        int r = charArray.length - 1;
        for (int i = charArray.length; i > 0; i--) {
            if (charArray[i - 1] != ' ') {
                r = i;
                break;
            }
        }
        return r - l - 1;
    }

    /**
     * 3. 无重复字符的最长子串
     *
     * @param s
     * @return
     */
    public int lengthOfLongestSubstring(String s) {
        char[] charArray = s.toCharArray();
        if (charArray.length <= 1) return charArray.length;
        int l = 0;
        int r = 1;
        int res = Integer.MIN_VALUE;
        int[] set = new int[256];
        set[charArray[0]] = 1;
        while (r < charArray.length) {
            while (set[charArray[r]] != 0 && l < r) {
                set[charArray[l++]] = 0;
            }
            res = Math.max(r - l + 1, res);
            set[charArray[r++]] = charArray[r - 1];
        }
        return res;
    }

    /**
     * 581. 最短无序连续子数组
     *
     * @param nums
     * @return
     */
    public int findUnsortedSubarray(int[] nums) {
        if (nums.length <= 0) return 0;
        int index = 0;

        List<Integer> list = new ArrayList<>();
        int minList = Integer.MAX_VALUE;
        for (int i = 1; i < nums.length; i++) {
            if (nums[index] <= nums[i]) {
                index = i;
            } else {
                list.add(i);
                minList = Math.min(nums[i], minList);
            }
        }
        if (list.size() == 0) {
            return 0;
        } else {
            int minIndex = list.get(0) - 1;
            while (minIndex >= 0 && nums[minIndex] > minList) {
                minIndex--;
            }
            minIndex++;
            return list.get(list.size() - 1) - minIndex + 1;
        }
    }

    /**
     * 301. 删除无效的括号
     *
     * @param s
     * @return
     */
    public List<String> removeInvalidParentheses(String s) {
        List<String> ans = new ArrayList<String>();
        Set<String> currSet = new HashSet<String>();

        currSet.add(s);
        while (true) {
            for (String str : currSet) {
                if (isOk(str)) {
                    ans.add(str);
                }
            }
            if (ans.size() > 0) {
                return ans;
            }
            Set<String> nextSet = new HashSet<String>();
            for (String str : currSet) {
                for (int i = 0; i < str.length(); i++) {
                    if (i > 0 && str.charAt(i) == str.charAt(i - 1)) {
                        continue;
                    }
                    if (str.charAt(i) == '(' || str.charAt(i) == ')') {
                        nextSet.add(str.substring(0, i) + str.substring(i + 1));
                    }
                }
            }
            currSet = nextSet;
        }
    }

    public Boolean isOk(String sb) {
        int cnt = 0;
        for (int i = 0; i < sb.length(); i++) {
            if (sb.charAt(i) == '(') {
                cnt++;
            } else if (sb.charAt(i) == ')') {
                cnt--;
                if (cnt < 0) {
                    return false;
                }
            }
        }
        return cnt == 0;
    }

    /**
     * 239. 滑动窗口最大值
     *
     * @param nums
     * @param k
     * @return
     */
    public int[] maxSlidingWindow(int[] nums, int k) {
        if (k == 1) return nums;
        int[] res = new int[nums.length - k + 1];
        int count = 0;
        int l = 0;
        int r = 1;
        Deque<Integer> deque = new ArrayDeque<>();//单调递减的单调队列,存放index,head最大
        deque.offerLast(l);
        while (r < k) {
            while (!deque.isEmpty() && nums[deque.peekLast()] <= nums[r]) {
                deque.pollLast();
            }
            deque.offerLast(r);
            r++;
        }

        l++;
        r++;
        res[count++] = nums[deque.peekFirst()];
        while (r <= nums.length) {
            if (l - 1 == deque.peekFirst()) {
                deque.pollFirst();
            }
            while (!deque.isEmpty() && nums[deque.peekLast()] <= nums[r - 1]) {
                deque.pollLast();
            }
            deque.offerLast(r - 1);
            l++;
            r++;
            res[count++] = nums[deque.peekFirst()];
        }
        return res;

        //另一个最初的版本
//        public int[] maxSlidingWindow(int[] nums, int k) {
//            if (k==1)return nums;
//            int[] res = new int[nums.length-k+1];
//            int count = 0;
//            int l = 0;
//            int r = 1;
//            int maxIndex = l;//表示一个窗口当中的最大的且最靠后的那个值的index
////        int secondMAXIndex = -1;//表示一个窗口当中的第二大的且最靠后的那个值的index
//            while (r<k){
//                if (nums[r]>=nums[maxIndex]){
//                    maxIndex = r;
//                }
//                r++;
//            }
//            l++;
//            r++;
//            res[count++] = nums[maxIndex];
//            while (r<=nums.length){
////            System.out.print("nums[l-1] = " + nums[l - 1]);//l-1表示被踢出去的那个值
////            System.out.println("  nums[r-1] = " + nums[r - 1]);//r-1表示新增的那个值
//                if (l-1==maxIndex){
//                    //如果之前的最大值被请出窗口了则再找一个新的
//                    maxIndex = l;
//                    for (int i = l; i < r; i++) {
//                        if (nums[i]>=nums[maxIndex]){
//                            maxIndex = i;
//                        }
//                    }
//                }
//
//                if (nums[r-1]>=nums[maxIndex]){
//                    maxIndex = r-1;
//                }
//                res[count++] = nums[maxIndex];
//                l++;
//                r++;
//            }
//            return res;
//        }
    }


    /**
     * 124. 二叉树中的最大路径和
     *
     * @param root
     * @return
     */
    public int maxPathSum(TreeNode root) {
        int[] ints = {Integer.MIN_VALUE};
        maxPathSum(root, ints);
        return ints[0];
    }

    public int maxPathSum(TreeNode root, int[] nums) {
        if (root == null) return 0;
        int leftRes = maxPathSum(root.left, nums);
        int rightRes = maxPathSum(root.right, nums);
        nums[0] = Math.max(nums[0], Math.max(Math.max(leftRes + rightRes + root.val, root.val), Math.max(leftRes, rightRes) + root.val));
        return Math.max(Math.max(leftRes, rightRes) + root.val, root.val);
    }

    /**
     * 85. 最大矩形
     *
     * @param matrix
     * @return
     */
    public int maximalRectangle(char[][] matrix) {
        int maxArea = 0;
        int[] heights = new int[matrix[0].length];
        Deque<Integer> deque = new ArrayDeque<>();
        for (int i = 0; i < matrix.length; i++) {
            //初始化heights数组
            for (int j = 0; j < heights.length; j++) {
                heights[j] = matrix[i][j] == '1' ? 1 + heights[j] : 0;
            }

            int[] left = new int[heights.length];//所能达到的最远位置但不包括
            int[] right = new int[heights.length];
            Arrays.fill(right, right.length);
            //维护一个单调递减的单调栈
            for (int j = 0; j < heights.length; j++) {
                while (!deque.isEmpty() && heights[j] <= heights[deque.peekLast()]) {
                    Integer l = deque.pollLast();
                    right[l] = j;
                }
                left[j] = deque.isEmpty() ? -1 : deque.peekLast();
                deque.offerLast(j);
            }
            deque.clear();
            for (int j = 0; j < heights.length; j++) {
                maxArea = Math.max((right[j] - left[j] - 1) * heights[j], maxArea);
            }
        }
        return maxArea;
    }

    /**
     * 32. 最长有效括号
     *
     * @param s
     * @return
     */
    public int longestValidParentheses(String s) {
        char[] charArray = s.toCharArray();
        Deque<Integer> deque = new ArrayDeque<>();
        for (int i = 0; i < charArray.length; i++) {
            if (!deque.isEmpty() && charArray[deque.peekLast()] == '(' && charArray[i] == ')') {
                deque.pollLast();
            } else {
                deque.offerLast(i);
            }
        }
        int res = 0;
        int preIndex = -1;
        while (!deque.isEmpty()) {
            Integer index = deque.pollFirst();
            res = Math.max(res, index - preIndex - 1);
            preIndex = index;
        }
        return Math.max(res, charArray.length - preIndex - 1);
    }

    /**
     * 10. 正则表达式匹配
     *
     * @param s
     * @param p
     * @return
     */
    public boolean isMatch2(String s, String p) {
        boolean[][] dp = new boolean[s.length() + 1][p.length() + 1];
        dp[0][0] = true;
        int m = s.length();
        int n = p.length();
        for (int i = 0; i <= m; ++i) {
            for (int j = 1; j <= n; ++j) {
                if (p.charAt(j - 1) == '*') {
                    dp[i][j] = dp[i][j - 2];
                    if (matches(s, p, i, j - 1)) {
                        dp[i][j] = dp[i][j] || dp[i - 1][j];
                    }
                } else {
                    if (matches(s, p, i, j)) {
                        dp[i][j] = dp[i - 1][j - 1];
                    }
                }
            }
        }
        return dp[s.length()][p.length()];
    }

    public boolean matches(String s, String p, int i, int j) {
        if (i == 0) {
            return false;
        }
        if (p.charAt(j - 1) == '.') {
            return true;
        }
        return s.charAt(i - 1) == p.charAt(j - 1);
    }

    /**
     * 89. 格雷编码
     *
     * @param n
     * @return
     */
    public List<Integer> grayCode(int n) {
        if (n == 0) {
            return List.of(0);
        }
        ArrayList<Integer> list = new ArrayList<>() {
            {
                add(0);
                add(1);
            }
        };
        if (n == 1) return list;
        grayCode(n, 2, list);
        return list;
    }

    public void grayCode(int max, int n, List<Integer> list) {
        int num = 1 << (n - 1);
        List<Integer> tempList = new ArrayList<>();
        for (Integer integer : list) {
            tempList.add(num | integer);
        }
        for (int i = tempList.size() - 1; i >= 0; i--) {
            list.add(tempList.get(i));
        }
        if (max == n) {
            return;
        } else {
            grayCode(max, n + 1, list);
        }
    }

    /**
     * 557. 反转字符串中的单词 III
     *
     * @param s
     * @return
     */
    public String reverseWords(String s) {
        char[] charArray = s.toCharArray();
        int preIndex = -1;
        for (int i = 0; i <= charArray.length; i++) {
            if (i == charArray.length || charArray[i] == ' ') {
                int l = preIndex + 1;
                int r = i - 1;
                while (l < r) {
                    char temp = charArray[l];
                    charArray[l] = charArray[r];
                    charArray[r] = temp;
                    l++;
                    r--;
                }
                preIndex = i;
            }
        }
        return String.valueOf(charArray);
    }

    /**
     * 292. Nim 游戏
     *
     * @param n
     * @return
     */
    public boolean canWinNim(int n) {
        return n % 4 != 0;
        //用动态规划的思想可以得出规律，被四整除的就返回false
//        if (n<=3)return true;
//        boolean a = true;
//        boolean b = true;
//        boolean c = true;
//        for (int i = 4; i <= n; i++) {
//            boolean tempRes = !a || !b || !c;
//            a = b;
//            b = c;
//            c = tempRes;
//            System.out.println(i+"  "+c);
//        }
//        return c;
    }

    /**
     * 43. 字符串相乘
     *
     * @param num1
     * @param num2
     * @return
     */
    public String multiply(String num1, String num2) {
        char[] num1CharArray = num1.toCharArray();
        char[] num2CharArray = num2.toCharArray();
        String res = "0";
        int countResTen = 0;
        for (int i = num1CharArray.length - 1; i >= 0; i--, countResTen++) {
            String tempRes = "0";
            int countTen = 0;
            for (int j = num2CharArray.length - 1; j >= 0; j--, countTen++) {
                String value = String.valueOf((num2CharArray[j] - '0') * (num1CharArray[i] - '0'));
                for (int z = 0; z < countTen; z++) {
                    value += '0';
                }
                tempRes = addStrings(value, tempRes);
            }
            for (int j = 0; j < countResTen; j++) {
                tempRes += '0';
            }
            res = addStrings(tempRes, res);
        }
        //去除先导0
        int p = 0;
        while (p < res.length() && res.charAt(p) == '0') {
            p++;
        }
        if (p == res.length()) p--;
        return res.substring(p);
    }

    public String addStrings(String num1, String num2) {
        StringBuilder builder = new StringBuilder();
        int carry = 0;
        for (int i = num1.length() - 1, j = num2.length() - 1; i >= 0 || j >= 0 || carry != 0; i--, j--) {
            int x = i < 0 ? 0 : num1.charAt(i) - '0';
            int y = j < 0 ? 0 : num2.charAt(j) - '0';
            int sum = (x + y + carry) % 10;
            builder.append(sum);
            carry = (x + y + carry) / 10;
        }
        return builder.reverse().toString();
    }

    /**
     * 231. 2 的幂
     *
     * @param n
     * @return
     */
    public boolean isPowerOfTwo(int n) {
//        if (n < 0) return false;
//        String binaryString = Integer.toBinaryString(n);
//        String substring = binaryString.substring(1);
//        return binaryString.charAt(0) == '1' && (substring.length() == 0 || Integer.parseInt(substring, 2) == 0);
        return n > 0 && ((n & (n - 1)) == 0);
    }


    /**
     * 0 1  -1(1->0)  2( 0 -> 1)
     * //原地更改，可以多定义几个状态
     * 289. 生命游戏
     *
     * @param board
     */
    public void gameOfLife(int[][] board) {
        int[] offset = new int[]{-1, 1, 0};
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[i].length; j++) {

                int liveNodeCount = 0;
                for (int k = 0; k < 3; k++) {
                    for (int l = 0; l < 3; l++) {

                        if (offset[k] == 0 && offset[l] == 0) {
                            continue;
                        }

                        int x = i + offset[k];
                        int y = j + offset[l];

                        if (x >= 0 && y >= 0 && x < board.length && y < board[i].length && Math.abs(board[x][y]) == 1) {
                            liveNodeCount++;
                        }

                    }
                }

                //更改状态
                if (liveNodeCount < 2 || liveNodeCount > 3) {
                    //死的还是死但是活的变死
                    if (board[i][j] == 1) {
                        board[i][j] = -1;
                    }
                } else if (liveNodeCount == 2) {
                    //状态保持不变
                } else if (liveNodeCount == 3) {
                    //死的变活，活的不变
                    if (board[i][j] == 0) {
                        board[i][j] = 2;
                    }
                }
            }

        }


        //最后更新状态
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[i].length; j++) {
                switch (board[i][j]) {
                    case -1 -> board[i][j] = 0;
                    case 2 -> board[i][j] = 1;
                }
            }
        }
    }


    /**
     * 328. 奇偶链表
     *
     * @param head
     * @return
     */
    public ListNode oddEvenList(ListNode head) {
        ListNode ji = new ListNode();
        ListNode ou = new ListNode();
        ListNode Jres = ji;
        ListNode Ores = ou;
        int index = 1;
        while (head != null) {
            if (index % 2 == 0) {
                ou.next = head;
                head = head.next;
                ou = ou.next;
                ou.next = null;
            } else {
                ji.next = head;
                head = head.next;
                ji = ji.next;
                ji.next = null;
            }
            index++;
        }
        ji.next = Ores.next;
        return Jres.next;
    }


    //    /**
//     * 29. 两数相除
//     * x^-1 == ~x 二进制取反
//     *
//     * @param
//     * @param
//     * @return
//     */
//    public int divide(int dividend, int divisor) {
//    }
    public int divide(int dividend, int divisor) {
        if (dividend == Integer.MIN_VALUE && divisor == -1) return Integer.MAX_VALUE;
        if (dividend == Integer.MIN_VALUE && divisor == 1) return Integer.MIN_VALUE;
        if (dividend > 0) return -divide(-dividend, divisor);
        if (divisor > 0) return -divide(dividend, -divisor);
        if (dividend > divisor) return 0;
        int res = 1, tmp = divisor;
        while ((dividend - divisor) <= divisor) {
            res += res;
            divisor += divisor;
        }
        return res + divide(dividend - divisor, tmp);
    }

    /**
     * 103. 二叉树的锯齿形层序遍历
     *
     * @param root
     * @return
     */
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {

        List<List<Integer>> res = new ArrayList<>();
        Deque<TreeNode> deque = new LinkedList<>();
        if (root != null) deque.offerLast(root);
        boolean order = true;
        while (!deque.isEmpty()) {
            int size = deque.size();
            List<Integer> list = new ArrayList<>();
            if (order) {
                while (size-- > 0) {
                    TreeNode node = deque.pollLast();
                    if (node.left != null) deque.offerFirst(node.left);
                    if (node.right != null) deque.offerFirst(node.right);
                    list.add(node.val);
                }
            } else {
                while (size-- > 0) {
                    TreeNode node = deque.pollFirst();
                    if (node.right != null) deque.offerLast(node.right);
                    if (node.left != null) deque.offerLast(node.left);
                    list.add(node.val);
                }
            }
            order = !order;
            res.add(list);
        }
        return res;
    }

    /**
     * 116. 填充每个节点的下一个右侧节点指针
     *
     * @param root
     * @return
     */
    public Node connect(Node root) {
        Deque<Node> deque = new LinkedList<>();
        if (root != null) deque.offerLast(root);
        while (!deque.isEmpty()) {

            Node c = deque.pollLast();
            Node t = c;

            int size = deque.size();
            while (size-- > 0) {
                Node node = deque.pollLast();
                c.next = node;
                c = c.next;
            }
            while (t != null) {
                if (t.left != null) deque.offerFirst(t.left);
                if (t.right != null) deque.offerFirst(t.right);
                t = t.next;
            }
            c.next = null;
        }
        return root;
    }

    /**
     * 138. 随机链表的复制
     *
     * @param head
     * @return
     */
    public Node copyRandomList(Node head) {
        List<Node> list = new ArrayList<>();
        Node h = head;
        Node copy = new Node(0);
        Node res = copy;
        while (head != null) {
            copy.next = new Node(head.val);
            head.val = list.size();
            list.add(copy.next);
            head = head.next;
            copy = copy.next;
        }

        copy = res;
        head = h;
        while (head != null) {
            if (head.random != null) copy.next.random = list.get(head.random.val);
            head = head.next;
            copy = copy.next;
        }

        return res.next;
    }

    /**
     * 140. 单词拆分 II
     *
     * @param s
     * @param wordDict
     * @return
     */
    public List<String> wordBreak(String s, List<String> wordDict) {
        HashSet<String> set = new HashSet<>(wordDict);
        ArrayList<String> res = new ArrayList<>();
        wordBreak(s, 0, set, new ArrayList<>(), res);
        return res;
    }

    public void wordBreak(String s, int i, HashSet<String> set, List<String> list, List<String> res) {
        for (int j = i + 1; j <= s.length(); j++) {
            String substring = s.substring(i, j);
            if (set.contains(substring)) {
                list.add(substring);
                if (j == s.length()) {
                    //如果到终点了则放入result中
                    StringBuilder add = new StringBuilder();
                    for (String a : list) {
                        add.append(a);
                        add.append(' ');
                    }
                    res.add(add.toString().trim());
                } else {
                    //没有到终点则继续递归
                    wordBreak(s, j, set, list, res);
                }
                //复原
                list.remove(list.size() - 1);
            }
        }
    }

    /**
     * 44. 通配符匹配
     *
     * @param s
     * @param p
     * @return
     */
    public boolean isMatch(String s, String p) {
        char[] pCharArray = p.toCharArray();
        char[] sCharArray = s.toCharArray();
        if (sCharArray.length == 0) {
            for (int i = 0; i < pCharArray.length; i++) {
                if (pCharArray[i] != '*') {
                    return false;
                }
            }
            return true;
        }
        if (pCharArray.length == 0) {
            return false;
        }
        boolean[][] dp = new boolean[sCharArray.length + 1][pCharArray.length + 1];
        dp[0][0] = true;
        for (int i = 1; i < dp[0].length && pCharArray[i - 1] == '*'; i++) {
            dp[0][i] = true;
        }
        for (int i = 1; i < dp.length; i++) {
            for (int j = 1; j < dp[i].length; j++) {
                if (pCharArray[j - 1] == '*') {
                    dp[i][j] = dp[i][j - 1] || dp[i - 1][j] || dp[i - 1][j - 1];  //空、字符
                } else if (pCharArray[j - 1] == '?' || pCharArray[j - 1] == sCharArray[i - 1]) {
                    dp[i][j] = dp[i - 1][j - 1];
                }
            }
        }
        return dp[sCharArray.length][pCharArray.length];
    }

    /**
     * 69. x 的平方根
     *
     * @param x
     * @return
     */
    public int mySqrt(int x) {
        if (x == Integer.MAX_VALUE) return 46340;
        if (x == 0) return 0;
        if (x <= 3) return 1;
        long l = 2;
        long r = (x + 1) / 2;
        while (l < r) {
            long mid = l + r + 1 >> 1;
            if (mid * mid <= x) {
                l = mid;
            } else {
                r = mid - 1;
            }
        }
        return (int) l;
    }

    /**
     * 91. 解码方法
     * 思路和爬楼梯有点类似
     * 写的时候边写边改，看着有点像屎山
     *
     * @param s
     * @return
     */
    public int numDecodings(String s) {
        char[] charArray = s.toCharArray();
        if (charArray.length == 0) {
            return 0;
        } else if (charArray.length == 1) {
            return charArray[0] == '0' ? 0 : 1;
        }
        int dp[] = new int[charArray.length];//该dp数组表示第i个之前的所有和,i由i-1与i-2得出
        //先初始化 dp[0]dp[1]
        if (charArray[0] == '0') {
            dp[0] = 0;
        } else {
            dp[0] = 1;
        }
        if (charArray[0] == '0') {
            dp[1] = 0;
        } else {
            boolean b = ((charArray[0] - '0') * 10 + charArray[1] - '0') <= 26;
            if (charArray[1] == '0') {
                if (b) {
                    dp[1] = 1;
                } else {
                    dp[1] = 0;
                }
            } else {
                if (b) {
                    dp[1] = 2;
                } else {
                    dp[1] = 1;
                }
            }
        }
        //再处理 i>=2的
        for (int i = 2; i < charArray.length; i++) {
            char c = charArray[i];
            char preC = charArray[i - 1];
            int temp1 = c - '0';
            int temp2 = (preC - '0') * 10 + temp1;

            if (c == '0' && preC == '0') {
                dp[i] = 0;
            } else if (c == '0') {
                if (temp2 <= 26) {
                    dp[i] = dp[i - 2];
                } else {
                    dp[i] = 0;
                }
            } else if (preC == '0') {
                dp[i] = dp[i - 1];
            } else {
                //前后两个字符都不为0
                if (temp2 > 26) {
                    dp[i] = dp[i - 1];
                } else {
                    dp[i] = dp[i - 1] + dp[i - 2];
                }
            }
        }
        return dp[charArray.length - 1];
    }

    /**
     * 162. 寻找峰值
     * 画出折线图观察，可得二分的规则
     *
     * @param nums
     * @return
     */
    public int findPeakElement(int[] nums) {
        int l = 0;
        int r = nums.length - 1;
        while (l < r) {
            int mid = l + r >> 1;
            int m = nums[mid];
            int mm = (mid + 1 >= nums.length) ? Integer.MIN_VALUE : nums[mid + 1];
            if (m >= mm) {
                r = mid;
            } else {
                l = mid + 1;
            }
        }
        return r;
    }


    /**
     * 149. 直线上最多的点数
     *
     * @param points
     * @return
     */
    public int maxPoints(int[][] points) {
        class MaxPointsLine {
            double k;
            double b;

            int state;

            public MaxPointsLine(double k, double b, int state) {
                this.k = k;
                this.b = b;
                this.state = state;
            }

            @Override
            public String toString() {
                final StringBuilder sb = new StringBuilder("MaxPointsLine{");
                sb.append("k=").append(k);
                sb.append(", b=").append(b);
                sb.append(", state=").append(state);
                sb.append('}');
                return sb.toString();
            }

            @Override
            public boolean equals(Object o) {
                if (this == o) return true;
                if (o == null || getClass() != o.getClass()) return false;

                MaxPointsLine line = (MaxPointsLine) o;

                if (Double.compare(line.k, k) != 0) return false;
                if (Double.compare(line.b, b) != 0) return false;
                return state == line.state;
            }

            @Override
            public int hashCode() {
                int result;
                long temp;
                temp = Double.doubleToLongBits(k);
                result = (int) (temp ^ (temp >>> 32));
                temp = Double.doubleToLongBits(b);
                result = 31 * result + (int) (temp ^ (temp >>> 32));
                result = 31 * result + state;
                return result;
            }
        }
        class MaxPointsDot {

            double x;
            double y;

            public MaxPointsDot(double x, double y) {
                this.x = x;
                this.y = y;
            }

            @Override
            public boolean equals(Object o) {
                if (this == o) return true;
                if (o == null || getClass() != o.getClass()) return false;

                MaxPointsDot that = (MaxPointsDot) o;

                if (Double.compare(that.x, x) != 0) return false;
                return Double.compare(that.y, y) == 0;
            }

            @Override
            public int hashCode() {
                int result;
                long temp;
                temp = Double.doubleToLongBits(x);
                result = (int) (temp ^ (temp >>> 32));
                temp = Double.doubleToLongBits(y);
                result = 31 * result + (int) (temp ^ (temp >>> 32));
                return result;
            }

            @Override
            public String toString() {
                final StringBuilder sb = new StringBuilder("MaxPointsDot{");
                sb.append("x=").append(x);
                sb.append(", y=").append(y);
                sb.append('}');
                return sb.toString();
            }
        }
        if (points.length <= 1) return points.length;
        Map<MaxPointsLine, Set<MaxPointsDot>> map = new HashMap<>();

        for (int i = 0; i < points.length; i++) {
            double x1 = points[i][0];
            double y1 = points[i][1];
            for (int j = 0; j < i; j++) {
                double x2 = points[j][0];
                double y2 = points[j][1];
                double k;
                double b;
                int state;
                if (y1 - y2 == 0) {
                    k = 0;
                    b = y1;
                    state = 1;
                } else if (x1 - x2 == 0) {
                    k = 1;
                    b = x1;
                    state = 2;
                } else {
                    k = (y1 - y2) / (x1 - x2);
                    b = y1 - k * x1;
                    state = 3;
                }
                MaxPointsLine line = new MaxPointsLine(k, b, state);
                Set<MaxPointsDot> set;
                if (!map.containsKey(line)) {
                    set = new HashSet<>();
                    map.put(line, set);
                } else {
                    set = map.get(line);
                }
                set.add(new MaxPointsDot(x1, y1));
                set.add(new MaxPointsDot(x2, y2));
            }
        }
        int res = Integer.MIN_VALUE;
        for (Set value : map.values()) {
            res = Math.max(value.size(), res);
        }
        return res;
    }

    /**
     * 166. 分数到小数
     *
     * @param numerator
     * @param denominator
     * @return
     */
    public String fractionToDecimal(int numerator, int denominator) {
        boolean fu = numerator < 0 ^ denominator < 0;
        long n = numerator;
        long d = denominator;
        n = Math.abs(n);
        d = Math.abs(d);
        StringBuilder sb = new StringBuilder();
        long i = n / d;//整数部分
        long remain = n % d;//余数部分
        sb.append(i);
        if (remain != 0) {
            Map<Long, Integer> map = new HashMap<>();
            sb.append('.');
            int index = sb.length();
            while (remain != 0) {
                if (map.containsKey(remain)) {
                    Integer preIndex = map.get(remain);
                    sb.insert(preIndex, "(");
                    sb.append(')');
                    break;
                }
                map.put(remain, index);
                remain *= 10;
                sb.append(remain / d);
                remain = remain % d;
                index++;
            }
        }
        if (fu && n != 0) {
            sb.insert(0, '-');
        }
        return sb.toString();
    }

    /**
     * 171. Excel 表列序号
     *
     * @param columnTitle
     * @return
     */
    public int titleToNumber(String columnTitle) {
        char[] charArray = columnTitle.toCharArray();
        int res = 0;
        for (int i = 0; i < charArray.length; i++) {
            res *= 26;
            int t = charArray[i] - 'A' + 1;
            res += t;
        }
        return res;
    }

    /***
     * 172. 阶乘后的零
     * 求5的因数
     * @param n
     * @return
     */
    public int trailingZeroes(int n) {
        if (n < 5) return 0;
        int a = 5;
        int count = 1;
        int i = n - 5;
        while (a <= i) {
            a += 5;
            int b = a;
            while (b % 5 == 0) {
                b /= 5;
                count++;
            }
        }
//        for (int i = 5; i <= n; i*=) {
//            int a = i;
//            int p = 0;
//            while (a > 0 && a % 5 == 0) {
//                a = a / 5;
//                p++;
//            }
//            res += p;
//        }
        return count;
    }

    /**
     * 190. 颠倒二进制位
     *
     * @param
     * @return
     */
    public int reverseBits(int n) {
        int res = 0;
        for (int i = 0; i < 32; i++) {
            if ((n & (1 << i)) != 0) {
                res = res | Integer.MIN_VALUE >>> i;
            }
        }
        return res;
    }

    /**
     * @param nums
     * @return
     */
    public String largestNumber(int[] nums) {
        List<String> collect = new ArrayList<>(Arrays.stream(nums).mapToObj(String::valueOf).toList());
        Collections.sort(collect, new Comparator<String>() {
            @Override
            public int compare(String s1, String s2) {
                //设定s1的length小于等于s2
                if (s1.length() > s2.length()) return -compare(s2, s1);
                if (s1.charAt(0) == s2.charAt(0) && s1.length() != s2.length()) {
                    return (s1 + s2).compareTo(s2 + s1);
                }
                return s1.compareTo(s2);
            }
        });
        StringBuilder sb = new StringBuilder();
        for (int i = collect.size() - 1; i >= 0; i--) {
            sb.append(collect.get(i));
        }
        //去除前导零
        int i = 0;
        while (i < sb.length() - 1) {
            if (sb.charAt(i) == '0') {
                i++;
            } else {
                break;
            }
        }
        sb.delete(0, i);
        return sb.toString();
    }

    /**
     * 191. 位1的个数
     * 逻辑右移
     *
     * @param n
     * @return
     */
    public int hammingWeight(int n) {
        int res = 0;
        for (int i = 1; i <= 32; i++) {
            int temp = n;
            temp = temp << (32 - i);
            temp = temp >>> (31);
            if (temp == 1) {
                res++;
            }
        }
        return res;
//        return Integer.bitCount(n);
    }

    /**
     * 210. 课程表 II
     * 拓扑排序、还需要判断是否有环
     *
     * @param numCourses
     * @param prerequisites
     * @return
     */
    public int[] findOrder(int numCourses, int[][] prerequisites) {
        Map<Integer, Set<Integer>> map = new HashMap<>();//存储图
        Map<Integer, Integer> rudu = new HashMap<>();//存储入度
        for (int i = 0; i < numCourses; i++) {
            map.put(i, new HashSet<>());
            rudu.put(i, 0);
        }
        for (int i = 0; i < prerequisites.length; i++) {
            int curr = prerequisites[i][0];
            int pre = prerequisites[i][1];
            Set<Integer> set = map.get(pre);
            set.add(curr);
            rudu.put(curr, rudu.get(curr) + 1);
        }
        boolean a = true;
        //判断是否有环
        for (Integer value : rudu.values()) {
            if (value == 0) {
                a = false;
                break;
            }
        }
        if (a) return new int[0];
        //构建完关系之后开始处理
        int index = 0;
        int[] res = new int[numCourses];
        while (map.size() > 0) {
            boolean hasZero = false;
            for (Map.Entry<Integer, Integer> entry : rudu.entrySet()) {
                if (map.size() <= 0) break;

                Integer key = entry.getKey();
                Integer value = entry.getValue();

                //入度为0
                if (value <= 0 && map.containsKey(key)) {
                    hasZero = true;
                    res[index++] = key;
                    Set<Integer> set = map.get(key);
                    //给这个去除的结点的所有的下一个结点的入度都减去1
                    if (set == null) {
                        return new int[0];
                    }
                    for (Integer next : set) {
                        rudu.put(next, rudu.getOrDefault(next, 1) - 1);
                    }
                    map.remove(key);//去除
                }
            }
            if (!hasZero) return new int[0];
        }
        return res;
    }


    /**
     * 212. 单词搜索 II
     *
     * @param board
     * @param words
     * @return
     */
    public List<String> findWords(char[][] board, String[] words) {
        StringBuilder sb = new StringBuilder();
        boolean[][] flag = new boolean[board.length][board[0].length];
        Trie trie = new Trie();
        for (String word : words) {
            trie.insert(word);
        }
        List<String> res = new ArrayList<>();
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[i].length; j++) {
                if (trie.set.isEmpty()) return res;
                flag[i][j] = true;
                sb.append(board[i][j]);
                findWords(board, words, res, sb, i, j, flag, trie);
                sb.deleteCharAt(sb.length() - 1);
                flag[i][j] = false;
            }
        }
        return res;
    }

    public void findWords(char[][] board, String[] words, List<String> res, StringBuilder sb, int x, int y, boolean[][] flag, Trie trie) {
        if (sb.length() > 10) return;
        if (trie.set.isEmpty()) return;
        String insert = sb.toString();
        if (trie.search(insert)) {
            trie.set.remove(insert);
            res.add(insert);
        }
        //如果是其中的一部分则继续查找
        if (!trie.startsWith(insert)) {
            return;
        }
        {
            int a = x - 1;
            int b = y;
            if (a >= 0 && b >= 0 && a < board.length && b < board[0].length && !flag[a][b]) {
                sb.append(board[a][b]);
                flag[a][b] = true;
                findWords(board, words, res, sb, a, b, flag, trie);
                flag[a][b] = false;
                sb.deleteCharAt(sb.length() - 1);
            }
        }
        {
            int a = x;
            int b = y - 1;
            if (a >= 0 && b >= 0 && a < board.length && b < board[0].length && !flag[a][b]) {
                sb.append(board[a][b]);
                flag[a][b] = true;
                findWords(board, words, res, sb, a, b, flag, trie);
                flag[a][b] = false;
                sb.deleteCharAt(sb.length() - 1);
            }
        }
        {
            int a = x + 1;
            int b = y;
            if (a >= 0 && b >= 0 && a < board.length && b < board[0].length && !flag[a][b]) {
                sb.append(board[a][b]);
                flag[a][b] = true;
                findWords(board, words, res, sb, a, b, flag, trie);
                flag[a][b] = false;
                sb.deleteCharAt(sb.length() - 1);
            }
        }
        {
            int a = x;
            int b = y + 1;
            if (a >= 0 && b >= 0 && a < board.length && b < board[0].length && !flag[a][b]) {
                sb.append(board[a][b]);
                flag[a][b] = true;
                findWords(board, words, res, sb, a, b, flag, trie);
                flag[a][b] = false;
                sb.deleteCharAt(sb.length() - 1);
            }
        }

    }

    /**
     * 227. 基本计算器 II
     *
     * @param s
     * @return
     */
    public int calculate(String s) {
        char[] charArray = s.toCharArray();
        int i = 0;
        //去除空格
        while (i < charArray.length && charArray[i] == ' ') {
            i++;
        }
        //先获取到第一个数字
        int num1 = 0;
        while (i < charArray.length && Character.isDigit(charArray[i])) {
            num1 *= 10;
            num1 += charArray[i] - '0';
            i++;
        }
        return calculate(charArray, i, num1);
    }

    //递归调用,num1表示前面一个操作数：  num1与num2的关系 num1 (+ - * /) num2
    public int calculate(char[] s, int i, int num1) {
        //去除空格
        while (i < s.length && s[i] == ' ') {
            i++;
        }
        //如果后面什么都没有就返回
        if (i >= s.length) return num1;
        //获得操作符
        char operation = s[i++];
        //去除空格
        while (i < s.length && s[i] == ' ') {
            i++;
        }
        //获取第二个操作数
        int num2 = 0;
        while (i < s.length && Character.isDigit(s[i])) {
            num2 *= 10;
            num2 += s[i] - '0';
            i++;
        }
        i--;
        int num3;
        //如果操作符号为乘法则直接计算，如果操作数为
        if (operation == '*' || operation == '/') {
            if (operation == '*') {
                num3 = num1 * num2;
            } else {
                num3 = num1 / num2;
            }
            num3 = calculate(s, i + 1, num3);
        } else {
            //如果是加减号则让后面的先运算完
            if (operation == '+') {
                num3 = num1 + calculate(s, i + 1, num2);
            } else {
                //如果是负号则传入 相反数 -num2
                num3 = num1 + calculate(s, i + 1, -num2);
            }
        }
        return num3;
    }

    /**
     * 268. 丢失的数字
     *
     * @param nums
     * @return
     */
    public int missingNumber(int[] nums) {
        int sum = 0;
        int r = 0;
        for (int i = 1; i <= nums.length; i++) {
            sum += i;
            r += nums[i - 1];
        }
        return sum - r;
    }

    /**
     * 341. 扁平化嵌套列表迭代器
     */
    interface NestedInteger {

        // @return true if this NestedInteger holds a single integer, rather than a nested list.
        public boolean isInteger();

        // @return the single integer that this NestedInteger holds, if it holds a single integer
        // Return null if this NestedInteger holds a nested list
        public Integer getInteger();

        // @return the nested list that this NestedInteger holds, if it holds a nested list
        // Return empty list if this NestedInteger holds a single integer
        public List<NestedInteger> getList();
    }

    class NestedIterator implements Iterator<Integer> {
        List<Integer> list;
        List<NestedInteger> nestedList;

        Iterator<Integer> me;

        public NestedIterator(List<NestedInteger> nestedList) {
            this.list = new ArrayList<>();
            this.nestedList = new ArrayList<>();
            buildNestedIterator(nestedList);
            this.me = list.iterator();
        }

        private void buildNestedIterator(List<NestedInteger> nestedList) {
            nestedList.forEach(nestedInteger -> {
                if (nestedInteger.isInteger()) {
                    list.add(nestedInteger.getInteger());
                } else {
                    buildNestedIterator(nestedInteger.getList());
                }
            });
        }

        @Override
        public Integer next() {
            return me.next();
        }

        @Override
        public boolean hasNext() {
            return me.hasNext();
        }
    }


    /**
     * 218. 天际线问题
     *
     * @param
     * @return
     */
    public List<List<Integer>> getSkyline(int[][] buildings) {
        List<int[]> list = new ArrayList<>();
        for (int[] building : buildings) {
            list.add(new int[]{building[0], building[2]});
            list.add(new int[]{building[1], -building[2]});
        }

        list.sort((arr1, arr2) -> {
            if (arr1[0] != arr2[0]) return Integer.compare(arr1[0], arr2[0]);
            else return Integer.compare(arr2[1], arr1[1]);//如果同一x值则height高的优先
        });
        //存放高度由大到小
        PriorityQueue<Integer> queue = new PriorityQueue<>((a, b) -> Integer.compare(b, a));
        List<List<Integer>> res = new ArrayList<>();
        int pre = 0;
        queue.add(0);
        for (int[] arr : list) {
            int index = arr[0];
            int height = arr[1];
            if (height > 0) {
                queue.add(height);
            } else {
                queue.remove(-height);
            }
            int cur = queue.peek();
            if (cur != pre) {
                res.add(List.of(index,cur));
                pre = cur;
            }
        }
        return res;
    }

    /**
     * 324. 摆动排序 II
     *
     * @param nums
     */
    public void wiggleSort(int[] nums) {
        wiggleSelect(nums, 0, nums.length - 1, nums.length >> 1);
        int x = nums[nums.length >> 1];
        int l = 0;
        int r = nums.length - 1;
        int index = 0;
        while (index <= r) {
            if (nums[index] < x) {
                int temp = nums[l];
                nums[l] = nums[index];
                nums[index] = temp;
                index++;
                l++;
            } else if (nums[index] > x) {
                int temp = nums[r];
                nums[r] = nums[index];
                nums[index] = temp;
                r--;
            } else {
                index++;
            }
        }
        int[] clone = nums.clone();
        index = clone.length - 1;
        for (int i = 1; i < nums.length; i += 2, index--) {
            nums[i] = clone[index];
        }
        for (int i = 0; i < nums.length; i += 2, index--) {
            nums[i] = clone[index];
        }
    }

    void wiggleSelect(int[] nums, int l, int r, int k) {
        if (l >= r) return;
        int i = l - 1;
        int j = r + 1;
        int p = nums[l + r >> 1];
        while (i < j) {
            do i++; while (nums[i] < p);
            do j--; while (nums[j] > p);
            if (i < j) {
                int temp = nums[i];
                nums[i] = nums[j];
                nums[j] = temp;
            }
        }
        if (k <= j) wiggleSelect(nums, l, j, k);
        else wiggleSelect(nums, j + 1, r, k);
    }


    /**
     * 排序
     *
     * @param nums
     * @return
     */
    public int[] sortArray(int[] nums) {
        digui(nums, 0, nums.length - 1);
        return nums;
    }

    void quickSort(int[] nums, int l, int r) {
        if (l >= r) return;
        int i = l - 1;
        int j = r + 1;
        int partition = nums[l + r >> 1];
        while (i < j) {
            do i++; while (nums[i] < partition);
            do j--; while (nums[j] > partition);
            if (i < j) {
                int temp = nums[i];
                nums[i] = nums[j];
                nums[j] = temp;
            }
        }
        quickSort(nums, l, j);
        quickSort(nums, j + 1, r);
    }

    void digui(int[] nums, int l, int r) {
        if (l >= r) return;
        int mid = l + r >> 1;
        digui(nums, l, mid);
        digui(nums, mid + 1, r);
        int i = l;
        int j = mid + 1;
        int index = 0;
        int[] arr = new int[r - l + 2];
        while (i <= mid && j <= r) {
            if (nums[i] < nums[j]) {
                arr[index++] = nums[i++];
            } else {
                arr[index++] = nums[j++];
            }
        }
        while (i <= mid) {
            arr[index++] = nums[i++];
        }
        while (j <= r) {
            arr[index++] = nums[j++];
        }
        System.arraycopy(arr, 0, nums, l, index);
    }


    public static void main(String[] args) {
        int[] nums = new int[]{2, 3, 1, 2, 2};
        int x = 1;
        int l = 0;
        int r = nums.length - 1;
        int index = 0;
        while (index <= r) {
            if (nums[index] < x) {
                int temp = nums[l];
                nums[l] = nums[index];
                nums[index] = temp;
                index++;
                l++;
            } else if (nums[index] > x) {
                int temp = nums[r];
                nums[r] = nums[index];
                nums[index] = temp;
                r--;
            } else {
                index++;
            }
        }
        System.out.println();
    }

}













