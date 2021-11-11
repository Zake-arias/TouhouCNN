# TouHou CNN
### 基于卷积神经网络的东方人物识别AI

## 快速开始

## 为我们提供训练数据集

> 想训练出一个识别准确的AI，大量的训练是必不可少的  
> 作为识别图片中东方Project人物的AI，  
> 为了给他提供给训练数据集，我们需要收集大量的东方人物图片  
> **这是个艰巨的任务** ~~其实是我是懒狗~~  

### 快速开始
1. 克隆项目  
    * 可以直接从本项目的GitHub仓库进行拉取`需要科学上网` 
    * 国内无法科学上网的车车人可以访问我们的[Gitee仓库](https://www.runoob.com)
2. 提交图片  
   > ## 注意！
   > 请勿在主分支提交R18图片  
   > 禁止提交非东方Project人物的图片  
   > 请正确为图片添加标签  
   >
   > ###### 为了避免造成不必要的麻烦，希望各位能够遵守相关规定，谢谢
   
   1. 打开<kbd>data\touhou</kbd>，这是存放训练用图片目录，请将准备好的图片放入此目录下  
   <br>
   2. 打开<kbd>data\touhou\index\index.csv</kbd>，这是用于存放数据集信息的索引文件，  
   结构如下:  
   
      ```python
       data/touhou/001.png,灵梦
       data/touhou/002.png,魔理沙
       data/touhou/037.png,露米娅
       data/touhou/038.png,大妖精
       #为了便于区分不同的角色，在这个CSV文件里，你可以使用类似python的'#'注释
       data/touhou/067.png,妹红
       data/touhou/068.png,萃香
       data/touhou/089.png,天子
       data/touhou/115.png,秦心
       ...
      ```  
      
      如你所见，这就是一个在普通不过的的CSV文件的格式，使用换行区分每一行，使用’,‘来区分每一列。  
      不过，为了方便大家区分不同角色，我们允许在CSV文件中使用类似Python的'#'进行注释。  
      当然，通常注释由我们预先添加，如角色名，请尽可能避免添加无意义的注释.  <br>  
      这种CSV格式的文件可以用市面上任意一种表格编辑工具打开，如Microsoft Excel。  
      当然，注释可能会被识别为单独的一行，也可能不会显示，但这并不影响文件正常使用。  
      如果你所使用的编辑器提示"使用CSV保存文件会损失部分功能"，请无视提示，继续使用CSV格式进行保存。  <br>
      1. 请按照上文示例代码中的格式进行索引的添加:
         
         ```
         文件路径,角色名
         #注释
         ```
         
         或者在表格编辑器中:

         | 文件路径 | 角色名 |  |
         | ------------ | ------------ | ------------ |
         | 文件路径 | 角色名 |  |
         | #一些注释 |  |  |

