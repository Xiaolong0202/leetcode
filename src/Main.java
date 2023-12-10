import java.io.*;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;

/**
 * @Author LiuXiaolong
 * @Description leetcode
 * @DateTime 2023/11/14  10:44
 **/
public class Main {

    public static void sumAB() throws IOException {
        StreamTokenizer streamTokenizer = new StreamTokenizer(new BufferedReader(new InputStreamReader(System.in)));
        while (streamTokenizer.nextToken() != -1) {
            int a = (int) streamTokenizer.nval;
            streamTokenizer.nextToken();
            int b = (int) streamTokenizer.nval;
            System.out.println(a + b);
        }
    }

    static void sumABTWO() throws Exception {
        StreamTokenizer in = new StreamTokenizer(new BufferedReader(new InputStreamReader(System.in)));
        in.nextToken();
        int n = (int) in.nval;
        while (n-- > 0) {
            in.nextToken();
            int a = (int) in.nval;
            in.nextToken();
            int b = (int) in.nval;
            System.out.println(a + b);
        }
    }

    static void sumAB3() throws Exception {
        PrintWriter out = new PrintWriter(new BufferedWriter(new OutputStreamWriter(System.out)));
        StreamTokenizer in = new StreamTokenizer(new BufferedReader(new InputStreamReader(System.in)));
        while (in.nextToken() != -1) {
            int a = (int) in.nval;
            in.nextToken();
            int b = (int) in.nval;
            if (a == 0 && b == 0) {
                out.flush();
                return;
            } else {
                out.println(a + b);
            }
        }
    }

    /**
     * 110. 平衡二叉树
     *
     * @param root
     * @return
     */
    boolean isBalancedRes = true;

    public boolean isBalanced(TreeNode root) {
        he(root);
        return isBalancedRes;
    }

    public int he(TreeNode root) {
        if (root == null) return 0;
        int l = he(root.left);
        int r = he(root.right);
        if (Math.abs(l - r) > 1) isBalancedRes = false;
        return Math.max(l, r) + 1;
    }

    /**
     * 129. 求根节点到叶节点数字之和
     *
     * @param root
     * @return
     */
    public int sumNumbers(TreeNode root) {
        return sumNumbers(root, 0);
    }

    public int sumNumbers(TreeNode node, int curSum) {
        if (node == null) {
            return 0;
        }
        int num = curSum * 10 + node.val;
        if (node.left == null && node.right == null) {
            return num;
        }
        return sumNumbers(node.left, num) + sumNumbers(node.right, num);
    }


    public static void main(String[] args) throws IOException, ParseException {
        SimpleDateFormat simpleDateFormat = new SimpleDateFormat("yyyy-MM-dd");
        Date parse = simpleDateFormat.parse("2023-12-11");
        System.out.println(parse);
    }
}
