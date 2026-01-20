; ModuleID = '__compute_module_part_03'
source_filename = "__compute_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@broadcast_add_fusion.2.clone_parallel_bounds = private unnamed_addr constant [18 x [2 x [2 x i64]]] [[2 x [2 x i64]] [[2 x i64] [i64 0, i64 1], [2 x i64] [i64 0, i64 1666]], [2 x [2 x i64]] [[2 x i64] [i64 0, i64 1], [2 x i64] [i64 1666, i64 3332]], [2 x [2 x i64]] [[2 x i64] [i64 0, i64 1], [2 x i64] [i64 3332, i64 4998]], [2 x [2 x i64]] [[2 x i64] [i64 0, i64 1], [2 x i64] [i64 4998, i64 6664]], [2 x [2 x i64]] [[2 x i64] [i64 0, i64 1], [2 x i64] [i64 6664, i64 8330]], [2 x [2 x i64]] [[2 x i64] [i64 0, i64 1], [2 x i64] [i64 8330, i64 10000]], [2 x [2 x i64]] [[2 x i64] [i64 1, i64 2], [2 x i64] [i64 0, i64 1666]], [2 x [2 x i64]] [[2 x i64] [i64 1, i64 2], [2 x i64] [i64 1666, i64 3332]], [2 x [2 x i64]] [[2 x i64] [i64 1, i64 2], [2 x i64] [i64 3332, i64 4998]], [2 x [2 x i64]] [[2 x i64] [i64 1, i64 2], [2 x i64] [i64 4998, i64 6664]], [2 x [2 x i64]] [[2 x i64] [i64 1, i64 2], [2 x i64] [i64 6664, i64 8330]], [2 x [2 x i64]] [[2 x i64] [i64 1, i64 2], [2 x i64] [i64 8330, i64 10000]], [2 x [2 x i64]] [[2 x i64] [i64 2, i64 3], [2 x i64] [i64 0, i64 1666]], [2 x [2 x i64]] [[2 x i64] [i64 2, i64 3], [2 x i64] [i64 1666, i64 3332]], [2 x [2 x i64]] [[2 x i64] [i64 2, i64 3], [2 x i64] [i64 3332, i64 4998]], [2 x [2 x i64]] [[2 x i64] [i64 2, i64 3], [2 x i64] [i64 4998, i64 6664]], [2 x [2 x i64]] [[2 x i64] [i64 2, i64 3], [2 x i64] [i64 6664, i64 8330]], [2 x [2 x i64]] [[2 x i64] [i64 2, i64 3], [2 x i64] [i64 8330, i64 10000]]]

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define noalias noundef ptr @broadcast_add_fusion.2.clone(ptr readonly captures(none) %0) local_unnamed_addr #0 {
  %tid_gep = getelementptr inbounds nuw i8, ptr %0, i64 8
  %tids = load ptr, ptr %tid_gep, align 8
  %tid_x = load i64, ptr %tids, align 4
  %args_gep = getelementptr inbounds nuw i8, ptr %0, i64 24
  %args = load ptr, ptr %args_gep, align 8
  %arg1_gep = getelementptr i8, ptr %args, i64 16
  %arg1 = load ptr, ptr %arg1_gep, align 8, !invariant.load !1, !dereferenceable !2, !align !3
  %lo_dim_0_gep = getelementptr inbounds [18 x [2 x [2 x i64]]], ptr @broadcast_add_fusion.2.clone_parallel_bounds, i64 0, i64 %tid_x, i64 0, i64 0
  %up_dim_0_gep = getelementptr inbounds [18 x [2 x [2 x i64]]], ptr @broadcast_add_fusion.2.clone_parallel_bounds, i64 0, i64 %tid_x, i64 0, i64 1
  %lo_dim_0 = load i64, ptr %lo_dim_0_gep, align 16
  %up_dim_0 = load i64, ptr %up_dim_0_gep, align 8
  %lo_dim_1_gep = getelementptr inbounds [18 x [2 x [2 x i64]]], ptr @broadcast_add_fusion.2.clone_parallel_bounds, i64 0, i64 %tid_x, i64 1, i64 0
  %up_dim_1_gep = getelementptr inbounds [18 x [2 x [2 x i64]]], ptr @broadcast_add_fusion.2.clone_parallel_bounds, i64 0, i64 %tid_x, i64 1, i64 1
  %lo_dim_1 = load i64, ptr %lo_dim_1_gep, align 16
  %up_dim_1 = load i64, ptr %up_dim_1_gep, align 8
  %.not5 = icmp ult i64 %lo_dim_0, %up_dim_0
  br i1 %.not5, label %broadcast_add_fusion.2.clone.loop_header.dim.1.preheader.lr.ph, label %return

broadcast_add_fusion.2.clone.loop_header.dim.1.preheader.lr.ph: ; preds = %1
  %arg0 = load ptr, ptr %args, align 8, !invariant.load !1, !dereferenceable !4, !align !3
  %.not13 = icmp ult i64 %lo_dim_1, %up_dim_1
  %2 = getelementptr inbounds nuw i8, ptr %arg0, i64 4
  %3 = load i32, ptr %2, align 4
  br i1 %.not13, label %broadcast_add_fusion.2.clone.loop_header.dim.1.preheader.us.preheader, label %return

broadcast_add_fusion.2.clone.loop_header.dim.1.preheader.us.preheader: ; preds = %broadcast_add_fusion.2.clone.loop_header.dim.1.preheader.lr.ph
  %broadcast.splatinsert10 = insertelement <8 x i32> poison, i32 %3, i64 0
  %broadcast.splat11 = shufflevector <8 x i32> %broadcast.splatinsert10, <8 x i32> poison, <8 x i32> zeroinitializer
  br label %broadcast_add_fusion.2.clone.loop_header.dim.1.preheader.us

broadcast_add_fusion.2.clone.loop_header.dim.1.preheader.us: ; preds = %broadcast_add_fusion.2.clone.loop_header.dim.1.preheader.us.preheader, %broadcast_add_fusion.2.clone.loop_header.dim.1.broadcast_add_fusion.2.clone.loop_exit.dim.1_crit_edge.us
  %broadcast_add_fusion.2.clone.invar_address.dim.0.06.us = phi i64 [ %invar.inc.us, %broadcast_add_fusion.2.clone.loop_header.dim.1.broadcast_add_fusion.2.clone.loop_exit.dim.1_crit_edge.us ], [ %lo_dim_0, %broadcast_add_fusion.2.clone.loop_header.dim.1.preheader.us.preheader ]
  %4 = mul i64 %broadcast_add_fusion.2.clone.invar_address.dim.0.06.us, 640000
  br label %vector.ph

vector.ph:                                        ; preds = %vector.ph, %broadcast_add_fusion.2.clone.loop_header.dim.1.preheader.us
  %broadcast_add_fusion.2.clone.invar_address.dim.1.04.us = phi i64 [ %lo_dim_1, %broadcast_add_fusion.2.clone.loop_header.dim.1.preheader.us ], [ %invar.inc3.us, %vector.ph ]
  %5 = shl i64 %broadcast_add_fusion.2.clone.invar_address.dim.1.04.us, 6
  %6 = add i64 %5, %4
  %broadcast.splatinsert = insertelement <8 x i64> poison, i64 %6, i64 0
  %broadcast.splat = shufflevector <8 x i64> %broadcast.splatinsert, <8 x i64> poison, <8 x i32> zeroinitializer
  %7 = trunc <8 x i64> %broadcast.splat to <8 x i32>
  %8 = or disjoint <8 x i32> %7, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %9 = trunc <8 x i64> %broadcast.splat to <8 x i32>
  %10 = or disjoint <8 x i32> %9, <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %11 = trunc <8 x i64> %broadcast.splat to <8 x i32>
  %12 = or disjoint <8 x i32> %11, <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23>
  %13 = trunc <8 x i64> %broadcast.splat to <8 x i32>
  %14 = or disjoint <8 x i32> %13, <i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %15 = add <8 x i32> %broadcast.splat11, %8
  %16 = add <8 x i32> %broadcast.splat11, %10
  %17 = add <8 x i32> %broadcast.splat11, %12
  %18 = add <8 x i32> %broadcast.splat11, %14
  %19 = getelementptr inbounds [3 x [10000 x [64 x i32]]], ptr %arg1, i64 0, i64 %broadcast_add_fusion.2.clone.invar_address.dim.0.06.us, i64 %broadcast_add_fusion.2.clone.invar_address.dim.1.04.us, i64 0
  %20 = getelementptr inbounds nuw i8, ptr %19, i64 32
  %21 = getelementptr inbounds nuw i8, ptr %19, i64 64
  %22 = getelementptr inbounds nuw i8, ptr %19, i64 96
  store <8 x i32> %15, ptr %19, align 64, !alias.scope !5
  store <8 x i32> %16, ptr %20, align 32, !alias.scope !5
  store <8 x i32> %17, ptr %21, align 64, !alias.scope !5
  store <8 x i32> %18, ptr %22, align 32, !alias.scope !5
  %23 = trunc <8 x i64> %broadcast.splat to <8 x i32>
  %24 = or disjoint <8 x i32> %23, <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39>
  %25 = trunc <8 x i64> %broadcast.splat to <8 x i32>
  %26 = or disjoint <8 x i32> %25, <i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47>
  %27 = trunc <8 x i64> %broadcast.splat to <8 x i32>
  %28 = or disjoint <8 x i32> %27, <i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55>
  %29 = trunc <8 x i64> %broadcast.splat to <8 x i32>
  %30 = or disjoint <8 x i32> %29, <i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  %31 = add <8 x i32> %broadcast.splat11, %24
  %32 = add <8 x i32> %broadcast.splat11, %26
  %33 = add <8 x i32> %broadcast.splat11, %28
  %34 = add <8 x i32> %broadcast.splat11, %30
  %35 = getelementptr inbounds [3 x [10000 x [64 x i32]]], ptr %arg1, i64 0, i64 %broadcast_add_fusion.2.clone.invar_address.dim.0.06.us, i64 %broadcast_add_fusion.2.clone.invar_address.dim.1.04.us, i64 32
  %36 = getelementptr inbounds nuw i8, ptr %35, i64 32
  %37 = getelementptr inbounds nuw i8, ptr %35, i64 64
  %38 = getelementptr inbounds nuw i8, ptr %35, i64 96
  store <8 x i32> %31, ptr %35, align 64, !alias.scope !5
  store <8 x i32> %32, ptr %36, align 32, !alias.scope !5
  store <8 x i32> %33, ptr %37, align 64, !alias.scope !5
  store <8 x i32> %34, ptr %38, align 32, !alias.scope !5
  %invar.inc3.us = add nuw nsw i64 %broadcast_add_fusion.2.clone.invar_address.dim.1.04.us, 1
  %exitcond8.not = icmp eq i64 %invar.inc3.us, %up_dim_1
  br i1 %exitcond8.not, label %broadcast_add_fusion.2.clone.loop_header.dim.1.broadcast_add_fusion.2.clone.loop_exit.dim.1_crit_edge.us, label %vector.ph, !llvm.loop !8

broadcast_add_fusion.2.clone.loop_header.dim.1.broadcast_add_fusion.2.clone.loop_exit.dim.1_crit_edge.us: ; preds = %vector.ph
  %invar.inc.us = add nuw nsw i64 %broadcast_add_fusion.2.clone.invar_address.dim.0.06.us, 1
  %exitcond9.not = icmp eq i64 %invar.inc.us, %up_dim_0
  br i1 %exitcond9.not, label %return, label %broadcast_add_fusion.2.clone.loop_header.dim.1.preheader.us, !llvm.loop !10

return:                                           ; preds = %broadcast_add_fusion.2.clone.loop_header.dim.1.broadcast_add_fusion.2.clone.loop_exit.dim.1_crit_edge.us, %broadcast_add_fusion.2.clone.loop_header.dim.1.preheader.lr.ph, %1
  ret ptr null
}

attributes #0 = { nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable "frame-pointer"="all" "prefer-vector-width"="256" }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"xla_dylib_index", i64 3}
!1 = !{}
!2 = !{i64 7680000}
!3 = !{i64 64}
!4 = !{i64 8}
!5 = !{!6}
!6 = !{!"result slice: {index:7, offset:7680064, size:7680000}", !7}
!7 = !{!"XLA host kernel broadcast_add_fusion.2.clone AA domain"}
!8 = distinct !{!8, !9}
!9 = !{!"llvm.loop.unroll.disable"}
!10 = distinct !{!10, !9}
