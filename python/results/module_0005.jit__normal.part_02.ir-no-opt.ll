; ModuleID = '__compute_module_part_02'
source_filename = "__compute_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%XLA_CPU_KernelCallFrame = type { ptr, ptr, i64, ptr }
%XLA_CPU_KernelThreadDim = type { i64, i64, i64 }
%XLA_CPU_KernelThread = type { i64, i64, i64 }
%XLA_CPU_KernelArg = type { ptr, i64 }

@__llvmsplit_unnamed.12 = private unnamed_addr constant [4 x i8] c"\01\00\00\00"
@__llvmsplit_unnamed.13 = private unnamed_addr constant [4 x i8] c" \00\00\00"
@broadcast_add_fusion.clone_parallel_bounds = private constant [18 x [2 x [2 x i64]]] [[2 x [2 x i64]] [[2 x i64] [i64 0, i64 1], [2 x i64] [i64 0, i64 1666]], [2 x [2 x i64]] [[2 x i64] [i64 0, i64 1], [2 x i64] [i64 1666, i64 3332]], [2 x [2 x i64]] [[2 x i64] [i64 0, i64 1], [2 x i64] [i64 3332, i64 4998]], [2 x [2 x i64]] [[2 x i64] [i64 0, i64 1], [2 x i64] [i64 4998, i64 6664]], [2 x [2 x i64]] [[2 x i64] [i64 0, i64 1], [2 x i64] [i64 6664, i64 8330]], [2 x [2 x i64]] [[2 x i64] [i64 0, i64 1], [2 x i64] [i64 8330, i64 10000]], [2 x [2 x i64]] [[2 x i64] [i64 1, i64 2], [2 x i64] [i64 0, i64 1666]], [2 x [2 x i64]] [[2 x i64] [i64 1, i64 2], [2 x i64] [i64 1666, i64 3332]], [2 x [2 x i64]] [[2 x i64] [i64 1, i64 2], [2 x i64] [i64 3332, i64 4998]], [2 x [2 x i64]] [[2 x i64] [i64 1, i64 2], [2 x i64] [i64 4998, i64 6664]], [2 x [2 x i64]] [[2 x i64] [i64 1, i64 2], [2 x i64] [i64 6664, i64 8330]], [2 x [2 x i64]] [[2 x i64] [i64 1, i64 2], [2 x i64] [i64 8330, i64 10000]], [2 x [2 x i64]] [[2 x i64] [i64 2, i64 3], [2 x i64] [i64 0, i64 1666]], [2 x [2 x i64]] [[2 x i64] [i64 2, i64 3], [2 x i64] [i64 1666, i64 3332]], [2 x [2 x i64]] [[2 x i64] [i64 2, i64 3], [2 x i64] [i64 3332, i64 4998]], [2 x [2 x i64]] [[2 x i64] [i64 2, i64 3], [2 x i64] [i64 4998, i64 6664]], [2 x [2 x i64]] [[2 x i64] [i64 2, i64 3], [2 x i64] [i64 6664, i64 8330]], [2 x [2 x i64]] [[2 x i64] [i64 2, i64 3], [2 x i64] [i64 8330, i64 10000]]]

; Function Attrs: uwtable
define ptr @broadcast_add_fusion.clone(ptr %0) #0 {
  %broadcast_add_fusion.clone.invar_address.dim.2 = alloca i64, align 8
  %broadcast_add_fusion.clone.invar_address.dim.1 = alloca i64, align 8
  %broadcast_add_fusion.clone.invar_address.dim.0 = alloca i64, align 8
  %tdims_gep = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 0
  %tdims = load ptr, ptr %tdims_gep, align 8
  %tdim_x_gep = getelementptr inbounds nuw %XLA_CPU_KernelThreadDim, ptr %tdims, i32 0, i32 0
  %tdim_y_gep = getelementptr inbounds nuw %XLA_CPU_KernelThreadDim, ptr %tdims, i32 0, i32 1
  %tdim_z_gep = getelementptr inbounds nuw %XLA_CPU_KernelThreadDim, ptr %tdims, i32 0, i32 2
  %tdim_x = load i64, ptr %tdim_x_gep, align 4
  %tdim_y = load i64, ptr %tdim_y_gep, align 4
  %tdim_z = load i64, ptr %tdim_z_gep, align 4
  %tid_gep = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 1
  %tids = load ptr, ptr %tid_gep, align 8
  %tid_x_gep = getelementptr inbounds nuw %XLA_CPU_KernelThread, ptr %tids, i32 0, i32 0
  %tid_y_gep = getelementptr inbounds nuw %XLA_CPU_KernelThread, ptr %tids, i32 0, i32 1
  %tid_z_gep = getelementptr inbounds nuw %XLA_CPU_KernelThread, ptr %tids, i32 0, i32 2
  %tid_x = load i64, ptr %tid_x_gep, align 4
  %tid_y = load i64, ptr %tid_y_gep, align 4
  %tid_z = load i64, ptr %tid_z_gep, align 4
  %args_gep = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args = load ptr, ptr %args_gep, align 8
  %arg0_gep = getelementptr %XLA_CPU_KernelArg, ptr %args, i32 0, i32 0
  %arg0 = load ptr, ptr %arg0_gep, align 8, !invariant.load !1, !dereferenceable !2, !align !3
  %args_gep1 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args2 = load ptr, ptr %args_gep1, align 8
  %arg1_gep = getelementptr %XLA_CPU_KernelArg, ptr %args2, i32 1, i32 0
  %arg1 = load ptr, ptr %arg1_gep, align 8, !invariant.load !1, !dereferenceable !4, !align !3
  %args_gep3 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args4 = load ptr, ptr %args_gep3, align 8
  %arg2_gep = getelementptr %XLA_CPU_KernelArg, ptr %args4, i32 2, i32 0
  %arg2 = load ptr, ptr %arg2_gep, align 8, !invariant.load !1, !dereferenceable !5, !align !3
  %args_gep5 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args6 = load ptr, ptr %args_gep5, align 8
  %arg3_gep = getelementptr %XLA_CPU_KernelArg, ptr %args6, i32 3, i32 0
  %arg3 = load ptr, ptr %arg3_gep, align 8, !invariant.load !1, !dereferenceable !5, !align !3
  %args_gep7 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args8 = load ptr, ptr %args_gep7, align 8
  %arg4_gep = getelementptr %XLA_CPU_KernelArg, ptr %args8, i32 4, i32 0
  %arg4 = load ptr, ptr %arg4_gep, align 8, !invariant.load !1, !dereferenceable !2, !align !3
  %args_gep9 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args10 = load ptr, ptr %args_gep9, align 8
  %arg5_gep = getelementptr %XLA_CPU_KernelArg, ptr %args10, i32 5, i32 0
  %arg5 = load ptr, ptr %arg5_gep, align 8, !invariant.load !1, !dereferenceable !5, !align !3
  %lo_dim_0_gep = getelementptr inbounds [18 x [2 x [2 x i64]]], ptr @broadcast_add_fusion.clone_parallel_bounds, i32 0, i64 %tid_x, i32 0, i32 0
  %up_dim_0_gep = getelementptr inbounds [18 x [2 x [2 x i64]]], ptr @broadcast_add_fusion.clone_parallel_bounds, i32 0, i64 %tid_x, i32 0, i32 1
  %lo_dim_0 = load i64, ptr %lo_dim_0_gep, align 4
  %up_dim_0 = load i64, ptr %up_dim_0_gep, align 4
  %lo_dim_1_gep = getelementptr inbounds [18 x [2 x [2 x i64]]], ptr @broadcast_add_fusion.clone_parallel_bounds, i32 0, i64 %tid_x, i32 1, i32 0
  %up_dim_1_gep = getelementptr inbounds [18 x [2 x [2 x i64]]], ptr @broadcast_add_fusion.clone_parallel_bounds, i32 0, i64 %tid_x, i32 1, i32 1
  %lo_dim_1 = load i64, ptr %lo_dim_1_gep, align 4
  %up_dim_1 = load i64, ptr %up_dim_1_gep, align 4
  store i64 %lo_dim_0, ptr %broadcast_add_fusion.clone.invar_address.dim.0, align 4
  br label %broadcast_add_fusion.clone.loop_header.dim.0

broadcast_add_fusion.clone.loop_header.dim.0:     ; preds = %broadcast_add_fusion.clone.loop_exit.dim.1, %1
  %broadcast_add_fusion.clone.indvar.dim.0 = load i64, ptr %broadcast_add_fusion.clone.invar_address.dim.0, align 4
  %2 = icmp uge i64 %broadcast_add_fusion.clone.indvar.dim.0, %up_dim_0
  br i1 %2, label %broadcast_add_fusion.clone.loop_exit.dim.0, label %broadcast_add_fusion.clone.loop_body.dim.0

broadcast_add_fusion.clone.loop_body.dim.0:       ; preds = %broadcast_add_fusion.clone.loop_header.dim.0
  store i64 %lo_dim_1, ptr %broadcast_add_fusion.clone.invar_address.dim.1, align 4
  br label %broadcast_add_fusion.clone.loop_header.dim.1

broadcast_add_fusion.clone.loop_header.dim.1:     ; preds = %broadcast_add_fusion.clone.loop_exit.dim.2, %broadcast_add_fusion.clone.loop_body.dim.0
  %broadcast_add_fusion.clone.indvar.dim.1 = load i64, ptr %broadcast_add_fusion.clone.invar_address.dim.1, align 4
  %3 = icmp uge i64 %broadcast_add_fusion.clone.indvar.dim.1, %up_dim_1
  br i1 %3, label %broadcast_add_fusion.clone.loop_exit.dim.1, label %broadcast_add_fusion.clone.loop_body.dim.1

broadcast_add_fusion.clone.loop_body.dim.1:       ; preds = %broadcast_add_fusion.clone.loop_header.dim.1
  store i64 0, ptr %broadcast_add_fusion.clone.invar_address.dim.2, align 4
  br label %broadcast_add_fusion.clone.loop_header.dim.2

broadcast_add_fusion.clone.loop_header.dim.2:     ; preds = %broadcast_add_fusion.clone.loop_body.dim.2, %broadcast_add_fusion.clone.loop_body.dim.1
  %broadcast_add_fusion.clone.indvar.dim.2 = load i64, ptr %broadcast_add_fusion.clone.invar_address.dim.2, align 4
  %4 = icmp uge i64 %broadcast_add_fusion.clone.indvar.dim.2, 64
  br i1 %4, label %broadcast_add_fusion.clone.loop_exit.dim.2, label %broadcast_add_fusion.clone.loop_body.dim.2

broadcast_add_fusion.clone.loop_body.dim.2:       ; preds = %broadcast_add_fusion.clone.loop_header.dim.2
  %5 = getelementptr inbounds [3 x [10000 x [64 x i32]]], ptr %arg3, i64 0, i64 %broadcast_add_fusion.clone.indvar.dim.0, i64 %broadcast_add_fusion.clone.indvar.dim.1, i64 %broadcast_add_fusion.clone.indvar.dim.2
  %6 = load i32, ptr %5, align 4, !invariant.load !1, !noalias !6
  %7 = getelementptr inbounds [3 x [10000 x [64 x i32]]], ptr %arg2, i64 0, i64 %broadcast_add_fusion.clone.indvar.dim.0, i64 %broadcast_add_fusion.clone.indvar.dim.1, i64 %broadcast_add_fusion.clone.indvar.dim.2
  %8 = load i32, ptr %7, align 4, !invariant.load !1, !noalias !6
  %9 = add i32 %6, %8
  %10 = getelementptr inbounds [3 x [10000 x [64 x i32]]], ptr %arg2, i64 0, i64 %broadcast_add_fusion.clone.indvar.dim.0, i64 %broadcast_add_fusion.clone.indvar.dim.1, i64 %broadcast_add_fusion.clone.indvar.dim.2
  %11 = load i32, ptr %10, align 4, !invariant.load !1, !noalias !6
  %12 = getelementptr inbounds [4 x i32], ptr %arg1, i64 0, i64 0
  %13 = load i32, ptr %12, align 4, !invariant.load !1, !noalias !6
  %14 = shl i32 %11, %13
  %shft.chk = icmp ult i32 %13, 32
  %15 = select i1 %shft.chk, i32 %14, i32 0
  %16 = getelementptr inbounds [3 x [10000 x [64 x i32]]], ptr %arg2, i64 0, i64 %broadcast_add_fusion.clone.indvar.dim.0, i64 %broadcast_add_fusion.clone.indvar.dim.1, i64 %broadcast_add_fusion.clone.indvar.dim.2
  %17 = load i32, ptr %16, align 4, !invariant.load !1, !noalias !6
  %constant.120 = load i32, ptr @__llvmsplit_unnamed.13, align 4
  %18 = sub i32 %constant.120, %13
  %19 = lshr i32 %17, %18
  %shft.chk13 = icmp ult i32 %18, 32
  %20 = select i1 %shft.chk13, i32 %19, i32 0
  %21 = or i32 %15, %20
  %22 = xor i32 %9, %21
  %23 = add i32 %9, %22
  %24 = getelementptr inbounds [4 x i32], ptr %arg1, i64 0, i64 1
  %25 = load i32, ptr %24, align 4, !invariant.load !1, !noalias !6
  %26 = shl i32 %22, %25
  %shft.chk14 = icmp ult i32 %25, 32
  %27 = select i1 %shft.chk14, i32 %26, i32 0
  %constant.12015 = load i32, ptr @__llvmsplit_unnamed.13, align 4
  %28 = sub i32 %constant.12015, %25
  %29 = lshr i32 %22, %28
  %shft.chk16 = icmp ult i32 %28, 32
  %30 = select i1 %shft.chk16, i32 %29, i32 0
  %31 = or i32 %27, %30
  %32 = xor i32 %23, %31
  %33 = add i32 %23, %32
  %34 = getelementptr inbounds [4 x i32], ptr %arg1, i64 0, i64 2
  %35 = load i32, ptr %34, align 4, !invariant.load !1, !noalias !6
  %36 = shl i32 %32, %35
  %shft.chk17 = icmp ult i32 %35, 32
  %37 = select i1 %shft.chk17, i32 %36, i32 0
  %constant.12018 = load i32, ptr @__llvmsplit_unnamed.13, align 4
  %38 = sub i32 %constant.12018, %35
  %39 = lshr i32 %32, %38
  %shft.chk19 = icmp ult i32 %38, 32
  %40 = select i1 %shft.chk19, i32 %39, i32 0
  %41 = or i32 %37, %40
  %42 = xor i32 %33, %41
  %43 = add i32 %33, %42
  %44 = getelementptr inbounds [4 x i32], ptr %arg1, i64 0, i64 3
  %45 = load i32, ptr %44, align 4, !invariant.load !1, !noalias !6
  %46 = shl i32 %42, %45
  %shft.chk20 = icmp ult i32 %45, 32
  %47 = select i1 %shft.chk20, i32 %46, i32 0
  %constant.12021 = load i32, ptr @__llvmsplit_unnamed.13, align 4
  %48 = sub i32 %constant.12021, %45
  %49 = lshr i32 %42, %48
  %shft.chk22 = icmp ult i32 %48, 32
  %50 = select i1 %shft.chk22, i32 %49, i32 0
  %51 = or i32 %47, %50
  %52 = xor i32 %43, %51
  %53 = load i32, ptr %arg0, align 4, !invariant.load !1, !noalias !6
  %54 = add i32 %52, %53
  %55 = load i32, ptr %arg4, align 4, !invariant.load !1, !noalias !6
  %constant.121 = load i32, ptr @__llvmsplit_unnamed.12, align 4
  %56 = add i32 %55, %constant.121
  %57 = add i32 %54, %56
  %58 = getelementptr inbounds [3 x [10000 x [64 x i32]]], ptr %arg5, i64 0, i64 %broadcast_add_fusion.clone.indvar.dim.0, i64 %broadcast_add_fusion.clone.indvar.dim.1, i64 %broadcast_add_fusion.clone.indvar.dim.2
  store i32 %57, ptr %58, align 4, !alias.scope !6
  %invar.inc12 = add nuw nsw i64 %broadcast_add_fusion.clone.indvar.dim.2, 1
  store i64 %invar.inc12, ptr %broadcast_add_fusion.clone.invar_address.dim.2, align 4
  br label %broadcast_add_fusion.clone.loop_header.dim.2

broadcast_add_fusion.clone.loop_exit.dim.2:       ; preds = %broadcast_add_fusion.clone.loop_header.dim.2
  %invar.inc11 = add nuw nsw i64 %broadcast_add_fusion.clone.indvar.dim.1, 1
  store i64 %invar.inc11, ptr %broadcast_add_fusion.clone.invar_address.dim.1, align 4
  br label %broadcast_add_fusion.clone.loop_header.dim.1, !llvm.loop !9

broadcast_add_fusion.clone.loop_exit.dim.1:       ; preds = %broadcast_add_fusion.clone.loop_header.dim.1
  %invar.inc = add nuw nsw i64 %broadcast_add_fusion.clone.indvar.dim.0, 1
  store i64 %invar.inc, ptr %broadcast_add_fusion.clone.invar_address.dim.0, align 4
  br label %broadcast_add_fusion.clone.loop_header.dim.0, !llvm.loop !11

broadcast_add_fusion.clone.loop_exit.dim.0:       ; preds = %broadcast_add_fusion.clone.loop_header.dim.0
  br label %return

return:                                           ; preds = %broadcast_add_fusion.clone.loop_exit.dim.0
  ret ptr null
}

attributes #0 = { uwtable "frame-pointer"="all" "prefer-vector-width"="256" }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"xla_dylib_index", i64 2}
!1 = !{}
!2 = !{i64 4}
!3 = !{i64 64}
!4 = !{i64 16}
!5 = !{i64 7680000}
!6 = !{!7}
!7 = !{!"result slice: {index:7, offset:7680064, size:7680000}", !8}
!8 = !{!"XLA host kernel broadcast_add_fusion.clone AA domain"}
!9 = distinct !{!9, !10}
!10 = !{!"llvm.loop.unroll.disable"}
!11 = distinct !{!11, !10}
