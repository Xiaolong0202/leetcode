import java.io.*;

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
    static void sumABTWO()throws Exception{
        StreamTokenizer in = new StreamTokenizer(new BufferedReader(new InputStreamReader(System.in)));
        in.nextToken();
        int n = (int)in.nval;
        while(n-->0){
            in.nextToken();
            int a = (int) in.nval;
            in.nextToken();
            int b = (int) in.nval;
            System.out.println(a+b);
        }
    }

    static void sumAB3() throws Exception{
        PrintWriter out = new PrintWriter(new BufferedWriter(new OutputStreamWriter(System.out)));
        StreamTokenizer in = new StreamTokenizer(new BufferedReader(new InputStreamReader(System.in)));
        while(in.nextToken()!=-1){
            int a = (int)in.nval;
            in.nextToken();
            int b = (int)in.nval;
            if(a==0&&b==0){
                out.flush();
                return;
            }else{
                out.println(a+b);
            }
        }
    }




    public static void main(String[] args) throws IOException {
        sumAB();
    }
}
