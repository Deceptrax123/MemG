from agent.memory import LocalGeometricDB, GeoLogicMemorySystem
import json
from collections import defaultdict
from agent.metrics import *
import threading
import concurrent.futures


class LocomoTester:
    def __init__(self, args, system, dataset_path):
        self.args = args
        self.system = system

        with open(dataset_path, 'r') as f:
            self.dataset = json.load(f)

    def parse_locomo_row_to_turns(self, sample):

        turns = []
        conv_data = sample['conversation']

        session_keys = [k for k in conv_data.keys() if k.startswith(
            "session_") and isinstance(conv_data[k], list)]
        session_keys.sort(key=lambda x: int(x.split('_')[1]))

        for key in session_keys:
            date_time_str = conv_data.get(f"{key}_date_time", "Unknown Time")

            for turn in conv_data[key]:
                text = turn.get("text", "")

                if "img_url" in turn and "blip_caption" in turn:
                    text = f"[Image: {turn['blip_caption']}] {text}"

                turns.append({
                    'speaker': turn.get("speaker", "Unknown"),
                    'date_time': date_time_str,
                    'text': text
                })

        return turns

    def run_test(self, limit=None, interval=3, resume_idx=0):
        samples = self.dataset[:interval]
        if resume_idx != 0:
            samples = self.dataset[resume_idx:resume_idx+interval]

        results = {
            "overall": {"f1": 0.0, "bleu": 0.0, "rouge": 0.0, "total": 0},
            "by_type": defaultdict(lambda: {"f1": 0.0, "bleu": 0.0, "rouge": 0.0, "total": 0}),
            "detailed_logs": []
        }

        new_idx = resume_idx
        for sample_idx, sample in enumerate(samples):
            print(
                f"\n{'='*60}\nTesting Conversation {sample_idx+1}/{len(samples)}\n{'='*60}")
            unique_conv_id = f"locomo_sample_{new_idx}"

            self.system.db.clear()

            if not self.system.db.load_session(unique_conv_id):
                print(f"[Ingesting New Conversation: {unique_conv_id}]")
                turns = self.parse_locomo_row_to_turns(sample)

                self.system.process_windows_parallel(
                    turns, window_size=self.args.window_size, overlap_size=self.args.overlap_size, max_workers=15)

                self.system.db.save_session(unique_conv_id)
            else:
                print(f"[Loaded Cached Memory for {unique_conv_id}]")

            current_time = time.time()

            for qa_idx, qa in enumerate(sample['qa']):
                question = qa['question']
                ground_truth = str(qa['answer']) if 'answer' in list(
                    qa.keys()) else str(qa['adversarial_answer'])
                q_type = str(qa.get('category'))

                try:
                    answer, rationale = self.system.answer_question(
                        question, top_k=self.args.top_k)

                    f1 = compute_f1(ground_truth, answer)
                    bleu = compute_bleu1(ground_truth, answer)
                    rouge = compute_rouge_l(ground_truth, answer)

                    results["overall"]["f1"] += f1
                    results["overall"]["bleu"] += bleu
                    results["overall"]["rouge"] += rouge
                    results["overall"]["total"] += 1

                    results["by_type"][q_type]["f1"] += f1
                    results["by_type"][q_type]["bleu"] += bleu
                    results["by_type"][q_type]["rouge"] += rouge
                    results["by_type"][q_type]["total"] += 1

                    results["detailed_logs"].append({
                        "conversation_id": unique_conv_id,
                        "question_id": qa_idx + 1,
                        "category": q_type,
                        "question": question,
                        "reference_answer": ground_truth,
                        "predicted_answer": answer,
                        "rationale": rationale,
                        "scores": {
                            "f1": round(f1, 2),
                            "bleu": round(bleu, 2),
                            "rouge": round(rouge, 2)
                        }
                    })

                    print(f"  Rationale : {rationale}")
                    print(f"  Predicted : {answer}")
                    print(f"  Reference : {ground_truth}")
                    print(
                        f"  Scores    : F1: {f1:.1f} | BLEU-1: {bleu:.1f} | ROUGE-L: {rouge:.1f}")

                except Exception as e:
                    print(f"\n Error processing Q{qa_idx+1}: {e}")

            new_idx += 1

        print("\n" + "="*50)
        print("🏆 FINAL EVALUATION RESULTS")
        print("="*50)

        total = results["overall"]["total"]
        if total > 0:
            print(f"Overall Performance (N={total}):")
            print(f"  F1      : {results['overall']['f1']/total:.2f}")
            print(f"  BLEU-1  : {results['overall']['bleu']/total:.2f}")
            print(f"  ROUGE-L : {results['overall']['rouge']/total:.2f}")

            print("\nPerformance by Question Category:")
            for q_type, stats in results["by_type"].items():
                t = stats["total"]
                if t > 0:
                    print(f"  Category [{q_type}] (N={t}):")
                    print(
                        f"    F1: {stats['f1']/t:.2f} | BLEU-1: {stats['bleu']/t:.2f} | ROUGE-L: {stats['rouge']/t:.2f}")

        try:
            with open(f"locomo_evaluation_details_{resume_idx}.json", "w") as f:
                json.dump(results["detailed_logs"], f, indent=4)
            print(
                "\nDetailed question-level results saved to 'locomo_evaluation_details.json'")
        except Exception as e:
            print(f"Could not save detailed results to file: {e}")
