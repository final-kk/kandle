- 当前由于webgpu dtype限制, 会在上传/下载数据时产生隐式dtype转换, 转换后的dtype忘记恢复
现在会导致逻辑dtype 和 物理dtype不符
- webgpu做inplce计算要执行临时分配, 需要想想有没有更好的解决方法.
- 即使做了scope / tidy, 还是存在显存泄漏, 需要进一步排查