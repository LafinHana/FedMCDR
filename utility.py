import json
def set_dataset(args):
    if args.full_overlap:
        if args.dataset == 'douban':
            with open('data/full_overlap/douban/domain_user_5.json', 'r') as f:
                domain_user = json.load(f)
                f.close()
            with open('data/full_overlap/douban/implicit_5.json', 'r') as f:
                dic = json.load(f)
                f.close()
            args.num_users = 1364
            args.num_domain = 3
            domain_names = ['Books', 'Movies', 'Music']
        else:
            if args.num_domain == 3:
                with open('data/full_overlap/3domains/domain_user.json', 'r') as f:
                    domain_user = json.load(f)
                    f.close()
                with open('data/full_overlap/3domains/implicit.json', 'r') as f:
                    dic = json.load(f)
                    f.close()
                args.num_users = 5539
                domain_names = ['Home', 'Books', 'Tools']
            if args.num_domain == 5:
                with open('data/full_overlap/5domains/domain_user.json', 'r') as f:
                    domain_user = json.load(f)
                    f.close()
                with open('data/full_overlap/5domains/implicit.json', 'r') as f:
                    dic = json.load(f)
                    f.close()
                args.num_users = 5539
                domain_names = ['Home', 'Books', 'Tools', 'Kindle', 'Clothing']
            if args.num_domain == 7:
                with open('data/full_overlap/7domains/domain_user.json', 'r') as f:
                    domain_user = json.load(f)
                    f.close()
                if args.simulate:
                    with open('data/full_overlap/7domains/sim_implicit.json', 'r') as f:
                        dic = json.load(f)
                        f.close()
                else:
                    with open('data/full_overlap/7domains/implicit.json', 'r') as f:
                        dic = json.load(f)
                        f.close()
                args.num_users = 5539
                domain_names = ['Home', 'Books', 'Tools', 'Kindle', 'Clothing', 'Electronics', 'Movies']
    else:
        if args.dataset == 'amazon':
            if args.num_domain == 4:
                with open('data/partial_overlap/4domains/domain_user.json', 'r') as f:
                    domain_user = json.load(f)
                    f.close()
                with open('data/partial_overlap/4domains/implicit.json', 'r') as f:
                    dic = json.load(f)
                    f.close()
                domain_names = ['Clothing', 'Books', 'Movies', 'CDs']
                args.num_users = 55518
            elif args.num_domain == 8:
                with open('data/partial_overlap/8domains/domain_user.json', 'r') as f:
                    domain_user = json.load(f)
                    f.close()
                with open('data/partial_overlap/8domains/implicit.json', 'r') as f:
                    dic = json.load(f)
                    f.close()
                domain_names = ['Clothing', 'Books', 'Home', 'Electronics', 'Sports', 'Cell', 'Movies', 'CDs']
                args.num_users = 98347
            else:
                with open('data/partial_overlap/16domains/domain_user.json', 'r') as f:
                    domain_user = json.load(f)
                    f.close()
                with open('data/partial_overlap/16domains/implicit.json', 'r') as f:
                    dic = json.load(f)
                    f.close()
                domain_names = ['Clothing', 'Books', 'Home', 'Electronics', 'Sports', 'Cell', 'Tools', 'CDs', 'Movies', 'Toys',
                                'Automotive', 'Pet', 'Kindle', 'Office', 'Patio', 'Grocery']
                args.num_users = 117672

        if args.dataset == 'douban':
            args.num_users = 2666
            domain_names = ['Book', 'Movie', 'Music']
            upath = 'data/douban_oldver/domain_user.json'
            dpath = 'data/douban_oldver/implicit.json'
            with open(upath, 'r') as f:
                domain_user = json.load(f)
                f.close()
            with open(dpath, 'r') as f:
                dic = json.load(f)
                f.close()

    return domain_user, dic, domain_names