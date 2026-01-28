/code/kandle-monorepo/packages/core/src/dispatch/handlers/factory.ts
实际上算是kernel了, 特别是pad, 在这里直接进行了计算
当前dispatch层 把 dispatch + native_functions的事情都做了, 这一层还是太厚了
拆分到2层在这里做 route => args => KernelParams ? 
统一的 KernelParams 看上去很诱惑? 但是工作量爆炸?