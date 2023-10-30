/**
 * @Author LiuXiaolong
 * @Description leetcode
 * @DateTime 2023/9/18  23:14
 **/
//这样可以自动帮我们生成equals hashcode toString JDK16 新特性,只能初始化一次，属性是final
public record TestRecord(String name ,String password) {
}
