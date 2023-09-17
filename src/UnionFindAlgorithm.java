/**
 * 并查集
 * @Author LiuXiaolong
 * @Description leetcode
 * @DateTime 2023/9/17  21:24
 **/
public class UnionFindAlgorithm {
    int father[] = new int[200000];

    public UnionFindAlgorithm() {
        for (int i = 0; i < father.length; i++) {
            father[i] = i;
        }
    }

    int find(int son){
        if (son == father[son]){
            return son;
        }
        return father[son] = find(father[son]);
    }

    void join(int u,int v){
        v =  find(v);
        u =  find(u);
       if (u==v)return;
       father[v] = u;
    }

}
