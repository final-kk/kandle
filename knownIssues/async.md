在js下由于语言特性, 几乎无法避免异步传染的情况
我只能尽可能的保证api的同步性. 特别是整个dispatch链路

已知的必要"异步触发"点
- 从webgpu后端tensor中获取数据
tensorFromWgpu.data() // -> Error("must call dataAsync() ")
await tensorFromWgpu.dataAsync() // -> ok

- Module.forward
由于混合多后端的存在, 无法保证forward的同步性
只能使用一个异步的forward, 因此在调用 Module时必须使用 await module.forward()
为了降低架构复杂度, 即使当前使用纯cpu推理, 也必须使用 await module.forward()
否则需要支持2套不同的api forwardSync/forwardAsync
由此会带来一些列的连锁影响, 因此仅使用命名 forward 且强制异步

需要额外注意的地方:
- 组合算子的数据读取
由于强制实行了前后端分离
默认的@kandle/core 没有任何的"计算能力"甚至tensor数据存储能力. 你可以把 pytorch 理解成 pytorch = userapi + cpubackend , 所以你可以可以直接torch.rand(), torch.zero(), 但是在我们的设计里, 并不想强制绑定一个cpu后端. 由此带来的问题是, 为了保证整个dispatch 链路的同步性, 大量的"组合算子" 实现复杂度变高, 因为不能直接在kernel中实现数据读取.
首次发现这种情况的kernel是为了audio模块中 biquad 的实现, biquad 显式的传入b0/b1/b2, a0/a1/a2 虽然他们是非常简单的标量或者tensor, 如果只是标量那么没有任何问题, 但是如果用户传入的是tensor. 由于计算架构的设计
一个tensor在计算过程中必须假定数据可能存在非cpu上,
因此直接读取数据可能是同步也可能是异步, 此时为了兼容所有情况, 比如会把 biquad 变成异步方法, 这就会导致异步传染直接上升到整个dispatch链路. 所以唯一的解法就是直接使用标量数字传入, 然后继续向后分发.因此部分api无法做到和torch的严格兼容, 用户必须手动前置处理数据同步问题. 并不支持tensor输入