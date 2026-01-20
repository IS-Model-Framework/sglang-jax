; ModuleID = '__compute_module_part_01'
source_filename = "__compute_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@bitcast_concatenate_fusion.1.clone_parallel_bounds = private unnamed_addr constant [5 x [1 x [2 x i64]]] [[1 x [2 x i64]] [[2 x i64] [i64 0, i64 2000]], [1 x [2 x i64]] [[2 x i64] [i64 2000, i64 4000]], [1 x [2 x i64]] [[2 x i64] [i64 4000, i64 6000]], [1 x [2 x i64]] [[2 x i64] [i64 6000, i64 8000]], [1 x [2 x i64]] [[2 x i64] [i64 8000, i64 10000]]]

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define noalias noundef ptr @bitcast_concatenate_fusion.1.clone(ptr readonly captures(none) %0) local_unnamed_addr #0 {
  %tid_gep = getelementptr inbounds nuw i8, ptr %0, i64 8
  %tids = load ptr, ptr %tid_gep, align 8
  %tid_x = load i64, ptr %tids, align 4
  %args_gep = getelementptr inbounds nuw i8, ptr %0, i64 24
  %args = load ptr, ptr %args_gep, align 8
  %arg0 = load ptr, ptr %args, align 8, !invariant.load !1, !dereferenceable !2, !align !3
  %arg1_gep = getelementptr i8, ptr %args, i64 16
  %arg1 = load ptr, ptr %arg1_gep, align 8, !invariant.load !1, !dereferenceable !4, !align !3
  %lo_dim_0_gep = getelementptr inbounds [5 x [1 x [2 x i64]]], ptr @bitcast_concatenate_fusion.1.clone_parallel_bounds, i64 0, i64 %tid_x, i64 0, i64 0
  %up_dim_0_gep = getelementptr inbounds [5 x [1 x [2 x i64]]], ptr @bitcast_concatenate_fusion.1.clone_parallel_bounds, i64 0, i64 %tid_x, i64 0, i64 1
  %lo_dim_0 = load i64, ptr %lo_dim_0_gep, align 16
  %up_dim_0 = load i64, ptr %up_dim_0_gep, align 8
  %.not2 = icmp ult i64 %lo_dim_0, %up_dim_0
  br i1 %.not2, label %vector.ph, label %return

vector.ph:                                        ; preds = %1, %vector.ph
  %bitcast_concatenate_fusion.1.clone.invar_address.dim.0.03 = phi i64 [ %invar.inc, %vector.ph ], [ %lo_dim_0, %1 ]
  %2 = getelementptr inbounds [3 x [10000 x [64 x float]]], ptr %arg0, i64 0, <8 x i64> zeroinitializer, i64 %bitcast_concatenate_fusion.1.clone.invar_address.dim.0.03, <8 x i64> <i64 0, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7>
  %wide.masked.gather = tail call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %2, i32 4, <8 x i1> splat (i1 true), <8 x float> poison), !invariant.load !1, !noalias !5
  %3 = getelementptr inbounds [10000 x [64 x float]], ptr %arg1, i64 0, i64 %bitcast_concatenate_fusion.1.clone.invar_address.dim.0.03, i64 0
  store <8 x float> %wide.masked.gather, ptr %3, align 64, !alias.scope !5
  %4 = getelementptr inbounds [3 x [10000 x [64 x float]]], ptr %arg0, i64 0, <8 x i64> zeroinitializer, i64 %bitcast_concatenate_fusion.1.clone.invar_address.dim.0.03, <8 x i64> <i64 8, i64 9, i64 10, i64 11, i64 12, i64 13, i64 14, i64 15>
  %wide.masked.gather.1 = tail call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %4, i32 4, <8 x i1> splat (i1 true), <8 x float> poison), !invariant.load !1, !noalias !5
  %5 = getelementptr inbounds [10000 x [64 x float]], ptr %arg1, i64 0, i64 %bitcast_concatenate_fusion.1.clone.invar_address.dim.0.03, i64 8
  store <8 x float> %wide.masked.gather.1, ptr %5, align 32, !alias.scope !5
  %6 = getelementptr inbounds [3 x [10000 x [64 x float]]], ptr %arg0, i64 0, <8 x i64> zeroinitializer, i64 %bitcast_concatenate_fusion.1.clone.invar_address.dim.0.03, <8 x i64> <i64 16, i64 17, i64 18, i64 19, i64 20, i64 21, i64 22, i64 23>
  %wide.masked.gather.2 = tail call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %6, i32 4, <8 x i1> splat (i1 true), <8 x float> poison), !invariant.load !1, !noalias !5
  %7 = getelementptr inbounds [10000 x [64 x float]], ptr %arg1, i64 0, i64 %bitcast_concatenate_fusion.1.clone.invar_address.dim.0.03, i64 16
  store <8 x float> %wide.masked.gather.2, ptr %7, align 64, !alias.scope !5
  %8 = getelementptr inbounds [3 x [10000 x [64 x float]]], ptr %arg0, i64 0, <8 x i64> splat (i64 1), i64 %bitcast_concatenate_fusion.1.clone.invar_address.dim.0.03, <8 x i64> <i64 24, i64 25, i64 26, i64 27, i64 28, i64 29, i64 30, i64 31>
  %wide.masked.gather.3 = tail call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %8, i32 4, <8 x i1> splat (i1 true), <8 x float> poison), !invariant.load !1, !noalias !5
  %9 = getelementptr inbounds [10000 x [64 x float]], ptr %arg1, i64 0, i64 %bitcast_concatenate_fusion.1.clone.invar_address.dim.0.03, i64 24
  store <8 x float> %wide.masked.gather.3, ptr %9, align 32, !alias.scope !5
  %10 = getelementptr inbounds [3 x [10000 x [64 x float]]], ptr %arg0, i64 0, <8 x i64> splat (i64 1), i64 %bitcast_concatenate_fusion.1.clone.invar_address.dim.0.03, <8 x i64> <i64 32, i64 33, i64 34, i64 35, i64 36, i64 37, i64 38, i64 39>
  %wide.masked.gather.4 = tail call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %10, i32 4, <8 x i1> splat (i1 true), <8 x float> poison), !invariant.load !1, !noalias !5
  %11 = getelementptr inbounds [10000 x [64 x float]], ptr %arg1, i64 0, i64 %bitcast_concatenate_fusion.1.clone.invar_address.dim.0.03, i64 32
  store <8 x float> %wide.masked.gather.4, ptr %11, align 64, !alias.scope !5
  %12 = getelementptr inbounds [3 x [10000 x [64 x float]]], ptr %arg0, i64 0, <8 x i64> <i64 1, i64 1, i64 1, i64 1, i64 2, i64 2, i64 2, i64 2>, i64 %bitcast_concatenate_fusion.1.clone.invar_address.dim.0.03, <8 x i64> <i64 40, i64 41, i64 42, i64 43, i64 44, i64 45, i64 46, i64 47>
  %wide.masked.gather.5 = tail call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %12, i32 4, <8 x i1> splat (i1 true), <8 x float> poison), !invariant.load !1, !noalias !5
  %13 = getelementptr inbounds [10000 x [64 x float]], ptr %arg1, i64 0, i64 %bitcast_concatenate_fusion.1.clone.invar_address.dim.0.03, i64 40
  store <8 x float> %wide.masked.gather.5, ptr %13, align 32, !alias.scope !5
  %14 = getelementptr inbounds [3 x [10000 x [64 x float]]], ptr %arg0, i64 0, <8 x i64> splat (i64 2), i64 %bitcast_concatenate_fusion.1.clone.invar_address.dim.0.03, <8 x i64> <i64 48, i64 49, i64 50, i64 51, i64 52, i64 53, i64 54, i64 55>
  %wide.masked.gather.6 = tail call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %14, i32 4, <8 x i1> splat (i1 true), <8 x float> poison), !invariant.load !1, !noalias !5
  %15 = getelementptr inbounds [10000 x [64 x float]], ptr %arg1, i64 0, i64 %bitcast_concatenate_fusion.1.clone.invar_address.dim.0.03, i64 48
  store <8 x float> %wide.masked.gather.6, ptr %15, align 64, !alias.scope !5
  %16 = getelementptr inbounds [3 x [10000 x [64 x float]]], ptr %arg0, i64 0, <8 x i64> splat (i64 2), i64 %bitcast_concatenate_fusion.1.clone.invar_address.dim.0.03, <8 x i64> <i64 56, i64 57, i64 58, i64 59, i64 60, i64 61, i64 62, i64 63>
  %wide.masked.gather.7 = tail call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %16, i32 4, <8 x i1> splat (i1 true), <8 x float> poison), !invariant.load !1, !noalias !5
  %17 = getelementptr inbounds [10000 x [64 x float]], ptr %arg1, i64 0, i64 %bitcast_concatenate_fusion.1.clone.invar_address.dim.0.03, i64 56
  store <8 x float> %wide.masked.gather.7, ptr %17, align 32, !alias.scope !5
  %invar.inc = add nuw nsw i64 %bitcast_concatenate_fusion.1.clone.invar_address.dim.0.03, 1
  %exitcond4.not = icmp eq i64 %invar.inc, %up_dim_0
  br i1 %exitcond4.not, label %return, label %vector.ph, !llvm.loop !8

return:                                           ; preds = %vector.ph, %1
  ret ptr null
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr>, i32 immarg, <8 x i1>, <8 x float>) #1

attributes #0 = { nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable "frame-pointer"="all" "prefer-vector-width"="256" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(read) }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"xla_dylib_index", i64 1}
!1 = !{}
!2 = !{i64 7680000}
!3 = !{i64 64}
!4 = !{i64 2560000}
!5 = !{!6}
!6 = !{!"result slice: {index:2, offset:0, size:2560000}", !7}
!7 = !{!"XLA host kernel bitcast_concatenate_fusion.1.clone AA domain"}
!8 = distinct !{!8, !9}
!9 = !{!"llvm.loop.unroll.disable"}
