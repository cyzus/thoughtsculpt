import numpy as np
import tqdm
from thoughtsculpt.model.utils import get_mask, OutlineSampler
def evaluate_find_least_interesting(ds, content_evaluator):
    correct = 0
    correct_start = 0
    correct_end = 0
    within_range = 0
    count = 0
    for i in range(len(ds)):
        data = ds[i]
        original_outlines = data["original_outlines"]
        mask = data["mask"]
        start = mask.index(0)
        if mask.count(0) == 1:
            end = start + 1
        else:
            end = mask[start+1:].index(0) + start + 1
        new_outlines = data["new_outlines"]
        interesting = data["interesting"]
        
        if not interesting:
            count += 1
            output = content_evaluator.predict_least_interesting_batched(new_outlines)
            if output == (start, end):
                correct += 1
            if output[0] >= start and output[1] <= end:
                within_range += 1
            if output[0] == start:
                correct_start += 1
            if output[1] == end:
                correct_end += 1
        

    accuracy = correct / count
    within_range_acc = within_range / count
    start_acc = correct_start / count
    end_acc = correct_end / count
    return {"accuracy": accuracy, "within_range_acc": within_range_acc, "start_acc": start_acc, "end_acc": end_acc}


def evaluate_interesting(ds, content_evaluator):
    correct = 0
    for i in tqdm.tqdm(range(len(ds))):
        data = ds[i]
        original_outlines = data["original_outlines"]
        mask = data["mask"]
        
        new_outlines = data["new_outlines"]
        interesting = data["interesting"]

        output = content_evaluator.predict(new_outlines, mask)
        if output == interesting:
            correct += 1
    return correct / len(ds)

def evaluate_interestiness_increase(ds, gpt_model, content_evaluator, gpt_evaluator=None):
    # interestiness_increases = []
    original_interestingesses = []
    new_interestingesses = []

    random_interestingnesses = []
    for data in tqdm.auto.tqdm(ds):
        original_outlines = data["original_outlines"]
        premise = data["premise"]
        original_interestingness = content_evaluator.predict_interestingness(original_outlines)
        outline_sampler = OutlineSampler(original_outlines, gpt_model, premise=premise)
        # indices found interestingness

        if gpt_evaluator is None:
            least_interesting_index, indices_list = content_evaluator.predict_least_interesting_batched(original_outlines, num_mask=2)
            
        else:
            least_interesting_index = gpt_evaluator.predict_least_interesting(original_outlines)
        prompt = outline_sampler.create_prompt(get_mask(len(original_outlines), *least_interesting_index), interested=True)
        interesting_outlines = outline_sampler.generate_new_outlines(prompt)
        interestingness = content_evaluator.predict_interestingness(interesting_outlines)
        # random indices
        random_mask = outline_sampler.sample_mask(num_mask=least_interesting_index[1] - least_interesting_index[0])
        prompt = outline_sampler.create_prompt(random_mask, interested=True)
        interesting_outlines = outline_sampler.generate_new_outlines(prompt)
        random_interestingness = content_evaluator.predict_interestingness(interesting_outlines)

        original_interestingesses.append(original_interestingness)
        new_interestingesses.append(interestingness)
        random_interestingnesses.append(random_interestingness)
    return{
        "original_interestingesses": np.mean(original_interestingesses),
        "new_interestingesses": np.mean(new_interestingesses),
        "random_new_interestingesses": np.mean(random_interestingnesses)
    }


def evaluate_score_increase(ds, content_evaluator, improver):
    original = []
    new = []
    origs = []
    news = []
    for data in tqdm.auto.tqdm(ds):
        original_outlines = data["original_outlines"]
        original_score = content_evaluator.evaluate_score(original_outlines)
        new_outlines = improver.improve(original_outlines)


        new_score = content_evaluator.evaluate_score(new_outlines)
        original.append(original_score)
        new.append(new_score)
        origs.append(original_outlines)
        news.append(new_outlines)
    return {
        "original":original,
        "new":new,
        "origs": origs,
        "news":news
    }

def compare_content(ds, comparer, improver1, improver2=None):
    result = []
    for data in tqdm.auto.tqdm(ds):
        original_outlines = data["original_outlines"]

        new_outlines1 = improver1.improve(original_outlines)
        if improver2 is not None:
            new_outlines2 = improver2.improve(original_outlines)
            pref = comparer.compare(new_outlines1, new_outlines2)
        else:
            pref = comparer.compare(original_outlines, new_outlines1)
        result.append(pref)
    return result



    
    
def improve_continuously(data, gpt_model, content_evaluator, step=3):

    original_outlines = data["original_outlines"]
    premise = data["premise"]
    original_interestingness = content_evaluator.predict_interestingness(original_outlines)
    outline_sampler = OutlineSampler(original_outlines, gpt_model, premise=premise)
    print("original interestingness:", original_interestingness)
    saved_outlines = [original_outlines]
    for i in range(step):
        outline_sampler.update_outlines(original_outlines)
        # indices found interestingness
        least_interesting_index, indices_list = content_evaluator.predict_least_interesting_batched(original_outlines, num_mask=2)
        print(f"least interesting index: {least_interesting_index}")
        prompt = outline_sampler.create_prompt(get_mask(len(original_outlines), *least_interesting_index), interested=True)

        interesting_outlines = outline_sampler.generate_new_outlines(prompt)
        interestingness = content_evaluator.predict_interestingness(interesting_outlines)
        original_outlines = interesting_outlines
        print(f"step {i+1}: {interestingness}")
        saved_outlines.append(interesting_outlines)
    return saved_outlines