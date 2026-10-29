# Title Defense Transcript

| Role | Name |
|---|---|
| Researchers | Mr. Lu, Mr. Almuena, Mr. Patricio, Mr. Ducyogen, Mr. Geronimo |
| Panel | Mr. Bautista, Mr. Cruz, Mr. Peralta |
| Research Advisor | Mr. Florentino |

**About this transcript**

- Clean verbatim. Fillers, stutters, and false starts are removed, and the wording is kept.
- Speech is in English and Filipino, as spoken.

---

**Mr. Lu:** Our research title is benchmarking multi-task optimizers for humanitarian classification and named entity recognition of Taglish typhoon tweets. When a typhoon crosses the Philippines, many Filipinos turn to social media to request help. Take this post for example: *"Baha na dito sa Marikina, need ng tubig para sa mga bata, please help."*

**Mr. Lu:** In our study, a model will do two tasks for a post like this. First is humanitarian classification. It takes the entire post and then identifies its intent and gives it a category. Next is named entity recognition, where it goes word by word through the post and identifies the key details.

**Mr. Lu:** A multitask model must do both tasks at the same time. The issue with this is that when a model does two tasks, one task may overpower the other during training, causing it to perform worse compared to when it is trained alone. This is called negative transfer. To address this issue, we have what is called a multitask optimizer. What it does is that it balances how each task influences the model during training.

**Mr. Lu:** Finally, for benchmarking, we will be taking 13 different optimizers that are trained under the same condition using the same dataset. First, we will be measuring how much negative transfer each optimizer has. And then continuing on from this, we will be measuring how much training time and memory they use. The final optimizer used on the final model will be selected based on which optimizer has the least amount of negative transfer and consumes the least amount of training time and memory.

**Mr. Almuena:** Currently, the Philippines has the highest disaster risk in the world. This is partially due to the fact that we're hit with 8 to 9 tropical cyclones every year. Now, in response to this, most Filipinos tend to turn to posting on social media as a means of asking for help. Now, first responders use the information gained from these posts. However, due to the sheer number of them, they're often overwhelmed, requiring them to turn to the use of automated models in order to help them sort through things.

**Mr. Almuena:** Now, models like these do exist, but based off the Philippine papers that we reviewed, most of these models have only been trained on a single task each. Now, models that are trained on multiple tasks do exist, like the one in Urdaneta City. However, this requires the use of a third-party LLM, such as ChatGPT, in order to work. An alternative method would be using a smaller model, such as a pre-trained encoder, in order for all of the processing to instead be done locally.

**Mr. Almuena:** Now, the problem with this is that each one of these tasks still requires a Taglish dataset. Now, while an English dataset does exist for this, none of them are specialized for Taglish specifically. Now, we do plan on making our own dataset, but even if we had one, we still run into a computational problem.

**Mr. Patricio:** As mentioned earlier, a model that was—

**Mr. Bautista:** Time out. Actually, let me interrupt you before you proceed. Can you explain the negative transfer in the simplest terms?

**Mr. Lu:** Negative transfer is when one task negatively impacts another. This is due to the task influencing the model to go in one direction, which negates the training that is done for the other.

**Mr. Lu:** Do you need more clarification on that po?

**Mr. Bautista:** Yeah, it doesn't sound clear. Can you please expound? In the simplest terms. Don't use any jargon.

**Mr. Lu:** Negative transfer is when one task tries to influence the model to learn in a certain way. What this does is that when another task tries to go another way, they will cancel each other out. Sometimes one task may do it more than the other, which means that it pulls the training in another direction opposite of what one task might want, causing it to reduce its performance compared to when it was trained alone.

**Mr. Bautista:** Okay, please continue.

**Mr. Cruz:** Wait a minute. I have some things to ask for your confirmation. So, the main function or purpose of your study is actually to combine the two, as you said a while ago, to combine the two models, right? Because right now, according to you, meron tayong mga models, but yung mga models na 'yan is trained individually. So, is it correct in my understanding that you're going to combine the two?

**Mr. Lu:** What we are doing is not necessarily combining two models. What we are doing is training one model to perform the two tasks at the same time.

**Mr. Cruz:** So, one model performing two tasks?

**Mr. Lu:** Yes po.

**Mr. Cruz:** And how many models do you have to train?

**Mr. Lu:** We have to train 13 models corresponding to the 13 different optimizers that we will be using.

**Mr. Cruz:** So each model will be trained twice? So magiging 26 na model, tama po ba 'yun? Am I...

**Mr. Lu:** No, each model will be trained thrice po.

**Mr. Cruz:** Thrice?

**Mr. Lu:** Yes, three times po.

**Mr. Cruz:** Three times yung each?

**Mr. Lu:** Yes po.

**Mr. Cruz:** So that is more than twice each. Now, you mentioned the optimizer a while ago, right?

**Mr. Lu:** Yes po.

**Mr. Cruz:** I'm thinking kung how fast will your optimizer be? Kasi ang concern ko dito is to deliver the news or the message correctly. 'Yun ang tingin ko ha, na merong konting correction na dapat doon sa message, right? Just like yung pinakita niyo a while ago. So, in correcting that message, since that is already an emergency, how fast is yung optimizer, yung correction na gagawin niyo before you actually release the corrected message?

**Mr. Lu:** Sorry, sir, can you repeat the question po?

**Mr. Cruz:** How fast is it for you to correct that message if ever there is something wrong with that message?

**Mr. Lu:** Are you stating that the model already made a mistake po?

**Mr. Cruz:** Yes. 'Yun ang pagkakaintindi ko, 'no? What you're trying to do here is to detect if there is some correction that should be done doon sa message before you actually release the message. Tama po ba 'yun?

**Mr. Almuena:** No po, sir. It's more like you get a post, and then you get to see yung humanitarian classification and then entity recognition po. It's more on trying to sort the posts and identify key aspects of it po, rather than straight up posting something.

**Mr. Cruz:** Kasi kanina iniisip ko, sabi ko, "Is there something wrong doon sa message?" 'Yun pala, you're going to use that as an example. Ayan: *"Baha na dito sa Marikina, need ng..."* So there's something wrong with *"need ng tubig,"* right? Tama ba?

**Mr. Lu:** No po. What this did is named entity recognition. It identified the key details within the post. It does not identify what is wrong. It only identifies what are the key details.

**Mr. Ducyogen:** So that instead of a human reading it, a computer will automatically assign a category so they don't need to manually read it or manually categorize each post po.

**Mr. Bautista:** So it's like an AI that tries to understand a Taglish post, getting the key relevant information like people, place, events? That's my understanding.

**Researchers:** Yes po.

**Mr. Bautista:** Wait, why 13? Why not 2? Why not 5? Why not 7? Where did you come up with the 13? You mentioned 13 optimizers, yung thesis niyo. I think that's the question of Sir Joel. Kasi it could be slower. If you just put 2 or whatever number, it could be faster, it could save on space or storage or overhead. So why 13? Where did you come up with the 13 optimizers?

**Mr. Patricio:** We need to test 13 optimizers because we need to find the optimizer that has the least negative transfer and the least training cost po. When we say training cost, this refers to the amount of time that's needed to train and the amount of memory it uses.

**Mr. Bautista:** Why 13? Why not 2?

**Mr. Ducyogen:** This is because Kurin et al. argued that using simpler methods or simpler optimizers is enough, rather than using advanced or more complex optimizers. But another paper, such as Gama and Grassi, argued that advanced optimizers have their own place depending on the problem. And the problem in this study is whether the multitask model can classify or can pick out the key details. And the reason it's 13 is because we picked out 6 from Kurin et al., then we picked out another method suggested by Ali et al., and other methods from Gama and Grassi, which is then combined into 13 optimizers.

**Mr. Cruz:** So you only based your 13 based on research?

**Mr. Ducyogen:** Yes po.

**Mr. Cruz:** Do you have any data to support yung 13 na 'yun, bukod sa ni-research niyo lang? Kasi pagka nag-research tayo, 'di ba, "Oo nga 'no, bakit 13 ang sinabi nito?" Then after that, probably yung curiosity mo is, "Bakit 13?" So you're going to go deeper, bakit 13. Probably you're going to look for more evidence, mga data. Do you have that one? Did you go deeper pa doon sa 13 na 'yun?

**Mr. Ducyogen:** Yes po. For example, one method was from Chai et al., which stated that their method works best on text-type datasets, while Kurin et al. used equal weighting on an image-type dataset. And another suggested optimizer was combining two optimizers, which was suggested by Ali et al.

**Mr. Cruz:** Suggested by... ano ba 'yun, foreigner? Tsaka yung study niya, saan naka-base? Meron bang specific place or event yung study na ginawa niya?

**Mr. Ducyogen:** That study from Ali et al. was—

**Mr. Cruz:** Kasi as you mentioned kanina, none for Taglish, right? So I'm thinking, saan niyo binase yung 13 niyo? According to you, based on the study. Kasi sino ba 'yun, author? She or he, a foreigner? Ano yung pinagbatayan niya? What specific place, country, or event, time and place? Kasi one of your goals is actually for Taglish, is it correct?

**Mr. Ducyogen:** Yes po.

**Mr. Cruz:** So iniisip ko yung study, saan nanggaling? Tapos right now you're going to apply it. Kung Taglish, ang first thing na papasok sa isip ko, you're going to apply it here in the Philippines, kasi Taglish, right? So saan kaya nanggaling yung research na 'yun? At bakit doon kayo nag-base sa research niya?

**Mr. Bautista:** Do you want to refer us to your document? Is there a page here where it says you used 13 because of this research? Can you refer to what page niyo nilagay?

**Mr. Peralta:** Or a specific study na nagsabi ng gano'n, and then another. Kasi 'pag isa lang study, dapat clear doon kung saan nga 'yung bansa, tapos ano yung ginamit. Kasi baka 'yung ni-refer niyong study, hindi applicable din sa atin dito. So dapat ma-justify niyo kasi ang sabi ni Sir, 'di ba, Taglish 'yung ina-ano niyo doon. So what specific study na nag-ano ng 13? You need to justify why you used 13. Kasi nagbanggit ka dito may nag-suggest ng dalawa. So ibig sabihin may dalawa, 'yung isa, ano 'yung isa? Is it kayo lang nag-decide na i-13 natin?

**Mr. Ducyogen:** No po.

**Mr. Peralta:** Oh, so may basis kayo. So ano nga ang basis, kaninong study 'yung 13 na 'yun na din ang ginamit niyo?

**Mr. Ducyogen:** It was from Xin et al. po.

**Mr. Peralta:** Anong ano niyo doon sa references ninyo? 13 ang ginamit niya?

**Mr. Ducyogen:** 11 po.

**Mr. Peralta:** 11, tapos in-add niyo lang 'yung 2 na suggested ng isa, that's why you came up with 13?

**Mr. Ducyogen:** Yes po. We added equal weighting into the study, which was argued to be...

**Mr. Peralta:** So ano 'yung method niyo na naka-come up kayo na kailangan i-add namin 'to? 11 plus 2, tapos i-13 na lang namin? Unless you're trying to create another model or another ano niyo na ite-test niyo na 13. Kumbaga anong tawag no'n, Sir? 'Yung sarili na nila, kumbaga in-adapt. Okay, that's why you have 13. Kasi nga, kung papel ang titingnan, ang mababasa lang sa papel, 'yung isa 11, 'yung isa dalawa. So ngayon ang study niyo 13. Panibago ba ito na gusto niyong may i-test tapos magkakaroon ng panibagong model? Or you just test the model lang?

**Mr. Patricio:** We test the optimizer po, and then we choose based on the result which will be used for the final model.

**Mr. Peralta:** Hindi, ano ang basis niyo pag-come up nga ng 13? Dapat ma-justify kasi doon sa paper na, "O sige, tatanggapin namin 13 ang ginamit." So dapat it is supported actually by an evidence, which is 'yung sa study. Otherwise, kailangan hanapan niyo talaga siya na ma-justify 'yung 13. Na-gets niyo ba 'yung question?

**Mr. Peralta:** Kagaya niyan sa mga IT, 'di ba, nagko-collect sila ng data. Ilan respondents nila? Iju-justify 'yun. Bakit? Ang justification eh ang number ng student na IT kunwari dito sa school—I'm giving an analogy para ma-gets ang question ni Sir, 'di ba? Kunwari may pinapa-evaluate sila, and then ang number of students na IT dito sa atin is sabihin nating 100. Ngayon, kumuha lang sila ng 20. So dapat supported 'yun. Supported 'yun dahil gumamit sila ng formula. So ang formula na 'yun, 'yung Slovin's formula or 'yung Taro Yamane formula. So justified na 20 lang ang kinuha.

**Mr. Peralta:** So in your case, 'yung na-mention ni Sir, hinahanap niya kaninong study na nag-refer ng 13 tapos may evidence talaga na itong 13 na 'to—lalo na sa inyo, 'di ba, optimizer, may model—so dapat talaga may pinanggagalingan, hindi 'yung gawa-gawa niyo lang sarili, unless may dine-develop kayong bago. Kasi maku-question 'yan 'pag ipe-present, 'di ba? We're planning also to present it in the university, 'di ba? So hindi lang kami ang magtatanong, kailangan hanapan niyo 'yun. Hindi kayo pwedeng mag-decide lang na, "Sige, 11 naman 'to tapos may dalawa isa, para sigurado na tayo, i-add na lang natin 13." Hindi kasi pwedeng gano'n. Meron ka dapat evidence na magsasabing ang standard ay 13 or sapat na ang 13 para makuha natin 'yung tamang result doon sa tine-test natin. So you have to provide that one. Or meron dito? Can you show us kung meron siya dito sa study na ituturo kami doon? Kasi ang daming references eh, hindi namin ma-isa-isa kung saang reference 'yung 13. Meron or wala? Para maka-proceed tayo, sabihin niyo lang. Kasi pwede naman namin siyang i-note na hanapan niyo ng ano, or you can change the number.

**Mr. Ducyogen:** In Chapter 2 po, we have a paper, Ali et al., which suggested to explore more optimizers than their study.

**Mr. Peralta:** Originally, ilan ang kanya?

**Mr. Ducyogen:** 2 po.

**Mr. Peralta:** 2 lang sa kanya, tapos nag-suggest siya sa recommendation niya ng mas madami pa?

**Mr. Ducyogen:** Yes po.

**Mr. Peralta:** Oo. Tapos ano pang study ang meron na may specific number siyang binigay?

**Mr. Ducyogen:** Another study was Gama and Grassi, which tested on 8 optimizers, because they know that those advanced optimizers will work on their harder dataset. And another was from Chai et al., which they tested on another two datasets which are different.

**Mr. Peralta:** Sa mga study ba nila, na-mention nila bakit specific na gano'n 'yung number nila? Kasi for sure, nung nag-study sila, ni-refer din nila 'yung study ng isa. So ano ang basis nila bakit tinaasan nila, and then to what certain limit nila itinaas? May standard ba or may formula bang sinusunod diyan, or may theory, or kung ano man diyan na para itaas ko to this extent? Kasi from 2, binanggit mo naging 8, so a difference of 6. Tapos from 8, may nabanggit kang 11, a difference of 3. So may mga certain ano ba na, 'pag itong dinagdag ko, ito din 'yung aasahan kong result? Level of significance na at 0.05, so ito 'yung result. Gano'n kasi sa IT eh. CS kasi kayo kaya medyo ina-ano ko lang. Meron bang standard na itong number na 'to, ito lang ang pagtaas para makuha natin 'yung ganitong percentage doon sa result natin? May gano'n ba doon, naka-mention ba sa kanila? Or they did not justify kung bakit nag-8 sila?

**Mr. Ducyogen:** For Xin et al., one of the reasons for that number was that the optimizers were grouped by their type.

**Mr. Bautista:** Sige, time na. To wrap up lang. So later, on top of the number, what's the criteria why you chose the 13? Like let's say there's 20 optimizers: one is faster, one is better, one is more accurate. First, why 13? Second, what's your criteria for choosing those 13 optimizers? "This one is faster, this one is accurate, that's why we chose the 13 optimizers." Is it supported by research, or by your own testing, or what?

**Mr. Peralta:** Walang gano'n, walang criteria? Or random lang?

**Mr. Ducyogen:** There was a criteria po, which was how they work.

**Mr. Bautista:** Wait. Is it written here in the document?

**Mr. Ducyogen:** Yes po.

**Mr. Bautista:** Can you refer it to us so...

**Mr. Ducyogen:** At the background of the study po.

**Mr. Bautista:** Sa background. So here it says you chose this optimizer because it is blah, blah, blah—it's faster, it is accurate, it uses less storage. So you have a criteria why you chose that optimizer, and you came up with 13?

**Mr. Ducyogen:** Yes po.

**Mr. Bautista:** Actually, I think they just based it sa research na binasa nila. And parang I don't see kasi where 'yung why you chose the 13. Kasi it's a basic question. Why not just 2?

**Mr. Ducyogen:** This is because Kurin et al., again, argued that this particular optimizer is better—

**Mr. Bautista:** Better in terms of... anyway, we can move on from that.

**Mr. Ducyogen:** ...including the other five methods they tested. Gama and Grassi also tested a total of 8, but they had different methods from Kurin et al., so we combined them.

**Mr. Bautista:** So our understanding is you based it from this research and you just came up with 13, and you're going to test. Maybe add information where you justify why you chose 13. You're going to add it later, because it's not here in the paper. So just to move on, we'll accept muna the 13 number for now, but take note of that question because we want to find out why you chose 13, and which particular optimizer.

**Mr. Cruz:** Take note din 'yung sinabi ni Ma'am. You chose 13, but meron bang applicable doon sa country, sa lugar? Very localized eh. Is it really applicable? 'Yun 'yung tanong ni Ma'am kanina.

**Mr. Ducyogen:** As far as I know, an optimizer is not specific to a place.

**Mr. Bautista:** Or language? Maybe the study was done in China, or in the US.

**Mr. Ducyogen:** The optimizers are agnostic to the data po. They rather interpret task gradients. Task gradients are signals, and those signals are numbers. So any images, numbers, and text are then converted into numbers. So it does not really matter what kind of input data you use, as it was stated by—

**Mr. Peralta:** So parang universal siya? Regardless kung anong language 'yan, basta itong optimizer na 'to, dumaan diyan, made-detect niya? So regardless of the language?

**Mr. Ducyogen:** Yes po.

**Mr. Peralta:** Pero sa study niyo, wala naman sa Pilipinas? Puro ibang bansa pa 'yung may study na gano'n? Most likely saang mga country 'to? 'Yung country ba na 'yun, bahain ba?

**Mr. Patricio:** Yes. Specifically, a study by Zou et al., they tested both of these tasks on Chinese typhoon posts.

**Mr. Peralta:** Sa Chinese, gaano kalala 'yung baha? Anong mga areas doon sa kanila bakit 'yun ang napili niyong study? Kasi mas better na parang i-copy dito sa atin kasi bahain din tayo.

**Mr. Cruz:** Meron bang nag-e-exist na dito sa Pilipinas?

**Mr. Peralta:** Wala daw sa Pilipinas, 'di ba, based on your study, wala pa? So first 'yan na study na Pilipinas 'yung ina-ano. Kasi sabi nga niya, Chinese. Ang Chinese ay bahain, tama? So, paano 'yung research niyo doon sa bahain sila? Lahat ba or ano?

**Mr. Ducyogen:** For our RRL po, rather than the area, they looked into tweets po.

**Mr. Peralta:** Sa tweets. Na-check niyo sa study niya na mas madaming gano'n 'yung tweets kaya marami ng dataset na nakuha?

**Mr. Ducyogen:** Yes po.

**Mr. Peralta:** Pero ang problem ko, Sir, 'di ba, saan niyo kukunin, Facebook?

**Mr. Ducyogen:** X po.

**Mr. Peralta:** Sa X lang, hindi sa Facebook?

**Mr. Ducyogen:** Both po. X first, and public Facebook pages if X does not have enough posts.

**Mr. Peralta:** Wala bang magiging issue 'yan, Sir, parang pagdating sa mga protocols or security? Kasi 'di ba kukuha kayo ng dataset doon? 'Yun 'yung magiging dataset ninyo, 'yung mga tweet na 'yun?

**Mr. Ducyogen:** For similar local studies po, such as Herrera et al. and Ermino et al., they anonymized the posts, and they argued that these posts were posted in public spaces. So before we use them, we have to anonymize them first. And we will use the same reasoning from those research that these posts were posted publicly.

**Mr. Peralta:** Hindi, paano niyo malalaman na 'yun ay pwede niyong kunin? Kasi 'di ba 'pag post lang 'yan, public, 'di ba, so po-post niya. So, paano niyo makukuha 'yung dataset na 'yun? Kasi 'di ba hindi ka naman basta-basta kukuha. Kahit nga i-share mo lang 'yung post, 'di ba, kailangan may credit ka doon sa baba, nilalagay mo "Credit to ganito," kasi may nakakasuhan 'pag hindi ka nagki-credit, 'di ba? Eh ito kasi, 'yun 'yung magiging dataset ninyo, 'di ba, para ma-test niyo 'tong sinasabi niyo.

**Mr. Peralta:** So naisip niyo na ba pagdating ng susunod na stage ng inyong research, paano niyo kukunin 'yung mga gano'n? Kasi 'di ba merong security 'yun na para makuha, or randomly kuha lang tayo nang kuha ng data? Paano natin malalaman na legit 'yung data na kukunin ninyo? Kasi hindi pwedeng screenshot lang eh, 'di ba, ipo-provide niyo dataset, ini-screenshot niyo lang mga gano'n, kasi parang kinuha mo lang din 'yung mga data. So paano magiging legit 'yung pagkuha?

**Mr. Ducyogen:** Ermino et al. or Herrera et al. used the official API and the post IDs.

**Mr. Peralta:** 'Yan, isa din 'yan na sabi niya na kumuha kayo ng parang post ID, 'di ba?

**Mr. Ducyogen:** Yes po.

**Mr. Peralta:** So paano niyo makukuha 'yun? Anong protocol ang hihingin diyan? Kasi pag-aari 'yan ng X, 'di ba? Paano niyo makukuha 'yun doon?

**Mr. Ducyogen:** According to the terms and agreement of X po, we can use them, but we cannot store the dataset.

**Mr. Peralta:** We can ano daw?

**Mr. Ducyogen:** We can use, but we cannot store it. So if we're going to use it for training, we will have to pay them for each post.

**Mr. Peralta:** Pay?

**Mr. Ducyogen:** Yes po.

**Mr. Peralta:** As in bayad, pay?

**Mr. Ducyogen:** Yes po.

**Mr. Peralta:** So every dataset na makukuha niyo, magbabayad kayo? Kanino kayo magbabayad, kay X or sa nag-post?

**Mr. Ducyogen:** Kay X po. Sa platform po mismo.

**Mr. Peralta:** So dapat ma-mention niyo 'yan dito sa document niyo, ma-expand niyo 'yung table. Kasi I've seen your table here, may nakalagay "Post ID," so dapat ma-expand niyo 'yon, 'yung protocol ng pagkuha niyo. Kasi para doon sa legitimacy ng dataset ninyo. Kasi 'yun 'yung ite-train, 'di ba? Doon manggagaling. So kailangan legit 'yung data niyo na kinuha, lalo na't kukuha kayo. Iniisip ko kasi kung ako nasa X tapos pinost ko 'yon, ako 'yung nag-post ng baha tapos binenta ng X sa inyo, 'di ba ako nag-post no'n pero hindi ko alam na ginamit na 'yung data. Is there a rule ba or policy sa X na once na pinost mo, pagmamay-ari na ng platform? Na-study niyo ba 'yon? Kasi kukuha kayo ng data doon eh, it's very important. Kagaya sa amin, 'di ba 'pag kumuha kami ng data, o 'pag gusto niyong magpa-interview, magpapa-pirma kayo kay Sir Joel o 'di kaya kay Ma'am Kristi para payagan kumuha ng data. The same also with this one, 'di ba sa X kayo kukuha. Paano 'yung pagkuha niyo, lalo na kukunin niyo 'yung Post ID? 'Di ba hindi naman 'yun publicly available? Kasi 'pag pinost mo, text lang talaga 'yan. Nandoon lang sa likod niyan, 'di ba, nakakaalam kayo niyan kasi may Post ID, the content, kung ano man 'yung format ng pag-save ni X ng post na 'yon.

**Mr. Peralta:** So ibig sabihin makikipag-communicate talaga kayo kay X to get that Post ID, otherwise parang hindi legit 'yung pagkuha niyo ng data. Although available siya public kasi publicly post siya eh, pero para kunin siya at anuhin doon, meron bang certain protocol na kailangan sundin?

**Mr. Peralta:** So that's why you need to expand your table here, mentioning the Post ID, anong protocol, paano niyo nakuha 'yon. Actually, meron nga sa data gathering eh, or kung paano 'yung paggamit niyo ng dataset. I-expand niyo 'yon, ipaliwanag niyo paano makukuha 'yung data. Kasi siyempre we're not just iniisip na i-present lang natin sa school, might as well after you graduate, ma-pursue niyo 'to, ma-publish, 'di ba? So siyempre maano natin 'yung data, saan ba 'to kumuha ng data? So legit dapat 'yung data, hindi 'yung laro-laro lang na basta kumuha. So dapat ma-mention niyo rito paano 'yung pagkuha, lalo na't Post ID pala 'yung kukunin ninyo. Na-gets niyo?

**Mr. Ducyogen:** Yes po.

**Mr. Peralta:** So dalawa na 'yon, 'yung sa 13 ni Sir at 'tsaka 'yung sa pagkuha ng Post ID. O sige, move on.

**Mr. Patricio:** As mentioned earlier, a model that performs both tasks can suffer performance at one of them, which is negative transfer. There are reasons to expect negative transfer with our two tasks. When Zhu et al. trained classification and Named Entity Recognition on English text, the tasks sent the model conflicting signals. Both tasks also scored lower compared to single-task models. Ali et al. argued that the Named Entity Recognition task overpowers the classification task because NER makes a decision per word, while classification only makes one per post.

**Mr. Patricio:** Multi-task optimizers are meant to reduce this negative transfer. However, some studies disagree on whether they help. Kurin et al. argued that a simple approach such as equal weighting, which treats both tasks as equally important, performed just as better as advanced optimizers. However, Gama and Grassi found that advanced optimizers did better on harder problems.

**Mr. Patricio:** The optimizers also differ in training cost. For instance, the advanced optimizers use about twice the memory and need more training time compared to a simple approach such as equal weighting.

**Mr. Patricio:** This brings us to the statement of the problem. How can one model perform both tasks while keeping the negative transfer and training cost low?

**Mr. Cruz:** "How can multi-task learning perform both tasks... on Taglish typhoon posts while keeping negative transfer and training cost low." So 'yun 'yung general statement of the problem?

**Mr. Geronimo:** The general objective of this study is we're going to develop a multi-task model that performs both of these tasks on Taglish typhoon posts.

**Mr. Cruz:** Sa development, ano 'yung gagamitin ninyo?

**Mr. Ducyogen:** Development of what po?

**Mr. Peralta:** This one, 'di ba you are creating a system, you're going to develop one?

**Mr. Ducyogen:** Yes po.

**Mr. Cruz:** I'm just curious kung ano 'yung gagamitin niyo na tool to proceed doon sa development.

**Mr. Ducyogen:** For the optimizers that we will be using in the study, we're going to use LibMTL po.

**Mr. Cruz:** You're going to use? Ano 'yun?

**Mr. Peralta:** MTL? Ano 'yun?

**Mr. Ducyogen:** It's a Python library po that allows us to implement the optimizers in an easier way compared to doing it all in raw PyTorch. And another one is using the Transformers library po for the model, because that's where we'll get the pre-trained encoder that we will be using.

**Mr. Cruz:** Ba't 'di niyo nilagay dito? But later, you'll discuss the...?

**Mr. Florentino:** Later po, meron, idi-discuss nila.

**Researchers:** Yes po.

**Mr. Geronimo:** For this study, we have four specific objectives. The first one is we're going to measure the single-task models' baseline scores. The second is we're going to design the multi-task model. The third, we're going to benchmark the 13 optimizer models. And lastly, we're going to compare the final multi-task model to the single-task models' baseline scores and training cost.

**Mr. Cruz:** If you're going to train that model, mag-aantay pa tayo ng sakuna?

**Mr. Peralta:** Hindi, sa post na nga daw nila kukunin, Sir, eh. Kaya tinatanong ko...

**Mr. Cruz:** Kahit 'yung mga previous posts?

**Mr. Peralta:** Hindi, dapat may specific na year na mga post. Kasi sabi ni Sir, kung ngayon, naghihintay ka muna ng may sakuna na bago gawin, 'di ba? So may specific dapat na kukunin, year 2025, from January to December...

**Mr. Patricio:** From 2020 to 2026 po.

**Mr. Peralta:** 2022 to 2026, ang dami ng taon. Maghahagilap kayo doon? Tapos specific sa X lang, 'di ba? Hindi sa Facebook at all?

**Mr. Patricio:** Yes po. If we don't meet the number sa X, we will use Facebook as a fallback.

**Mr. Peralta:** Medyo hesitant ako, Sir, kung paano niyo kukunin 'yung mga post. Or meron bang tinatawag, Sir, may nabasa lang ako, 'yung API, na pwedeng mapapayagan na maka-ano doon.

**Mr. Ducyogen:** 'Yung magbabayad sa X?

**Mr. Peralta:** Oo. Gusto kong malaman, kaya tinatanong ko kanina eh kung may ganoon na si X.

**Mr. Lu:** We will explain that later po.

**Mr. Peralta:** Kasi medyo hesitant ako na makakuha kayo ng data. Baka kasi iniisip niyo lang kunin lang sa post, 'di ba? Hindi kasi pwede 'yon, kuha lang nang kuha. Hindi kasi pwede 'yon. Dapat talaga meron kayong permission to get that. Although hindi malalaman ng kinuhaan niyo kasi nandito lang naman tayo sa study, pero kasi paano maaano 'yung data na legit 'yon? Na accurate 'yon? Baka kasi gawa-gawa lang, 'di ba? So dapat, sabi niyo may Post ID. So ang Post ID ba'y pinapayagan ba, may API ba 'yung X? Meron?

**Mr. Ducyogen:** Yes po.

**Mr. Peralta:** Walang payment 'yon, as in?

**Mr. Ducyogen:** Meron, it requires payment po.

**Mr. Peralta:** It requires. How much?

**Mr. Ducyogen:** $0.01 per post.

**Mr. Peralta:** Per post?

**Mr. Ducyogen:** Yes po.

**Mr. Peralta:** So kay X kayo makikipag-communicate no'n?

**Mr. Ducyogen:** Yes po.

**Mr. Peralta:** Then how many ang kukunin niyo? From 2022 to 2026? Pero hindi, gusto kong ma-justify bakit kailangan mag-2022 to 2026, eh may specific naman na buwan lang na may baha tapos doon lang. Bakit kailangan i-expand natin? Anong pinapatunayan niyo sa 2022 to 2026 kung ang babasahin lang naman 'yung Taglish na post, right? 'Di ba 'yun 'yung inaano niyo kanina, 'yung sa Taglish? So gusto kong malaman 'yung idea bakit kailangan pa nating magsimula sa 2022 tapos 2026? Anong pinapatunayan natin doon? Kung iisa lang naman ang motibo natin, 'yung makuha lang naman, ma-interpret 'yung Taglish, 'di ba, na post? Bakit kailangan mag-expand doon?

**Mr. Ducyogen:** This is because of Seeberger et al.

**Mr. Peralta:** Anong meron doon?

**Mr. Ducyogen:** They argued that the language evolves over time and that a model trained on older posts would perform slightly worse on newer ones.

**Mr. Peralta:** Anong kinukuha niyo doon? Anong pinapatunayan niyo? Speed, accuracy? Kasi 'yun 'yung nandoon sa number four, speed, accuracy. May kaibahan ba, nagiging mabilis ba, nagiging accurate kung luma at saka current 'yung kinuhaan na post? May pagbabago ba sa sentence construction na nag-post before at sa ngayon? Mas maalam na ba 'yung tao sa pagpo-post ng Taglish compared doon? So may ganoon bang sinasagot 'yung objective niyo or 'yung SOP niyo para doon?

**Mr. Peralta:** Kasi kagaya noon, dati Jejemon ang pagkaka-post sa Taglish. Kaya kumuha kayo ng 2022 kasi may pinapatunayan kayo na the year 2022, medyo ano pa 'yung mga tao sa pagpo-post ng ganito, ang gamit nila ganitong mga Tagalog pa. Hanggang sa 2023 medyo ano na. Ngayon kasi may mga "It's raining, hoy!" Oh, 'di ba? Mga ganoon na ang posting nila. So may pinapatunayan ba kayo na language na nag-iiba over time? Kasi kailangan niyong ma-justify bakit nagsimula tayo sa 2022 to 2026, na pwede naman sanang specific month na ang bahain sa Pilipinas, June, kasi tag-ulan, July. Kukuha tayo ng June 2026 as the latest, or June 2025 kung mas maraming baha noong year 2025. Ba't pa tayo umabot ng 2022? Kasi baka mas mahal na kung older. Kasi bayad 'to, 'di ba? So hahalungkatin pa ni X 'yon noong mga ganoong year, o baka mas mahirapan pa kayo noon sa pag-communicate with X. Kasi hindi naman basta-basta magre-respond sa'yo si X eh. Sino ba naman tayo para mag-respond agad si X sa atin, 'di ba, knowing that it's a study. 'Di ba? Kasi magpe-pay kayo eh, ganoon ba ka-responsive sila? So dapat pag-isipan niyo.

**Mr. Peralta:** Pwede naman specific na year or buwan. Kasi may specific naman tayo na calamity na pino-focus. So bakit natin kailangan? Can you give us some ano? Baka kasi meron kayong ibang naisip.

**Mr. Ducyogen:** It was a cost-saving measure na rin po. Because before 2023, it was free to access the...

**Mr. Peralta:** Noong 2023?

**Mr. Ducyogen:** Yes.

**Mr. Peralta:** Tapos?

**Mr. Ducyogen:** Then after that year, they no longer offer it for free. You must now agree to the terms and agreements, then you must now pay...

**Mr. Peralta:** You must now pay to get the data?

**Mr. Ducyogen:** Yes po.

**Mr. Peralta:** So doon tayo sa logic mo na free kasi noong 2023, tapos sumunod na taon, 2024, 25, 26, hindi na libre. So kung ang point mo lang ay makakuha tayo ng libre, so specific noong 2023 tayo kukuha kasi libre eh. Bakit tayo kukuha pa ng 2024, 2025 na post kung iiwas lang tayo doon sa pagbabayad? Pero kung gusto niyo naman pala magbayad, why not the latest? Kasi nga latest, current, mas maganda ang research. Kasi 'pag tiningnan natin 'yung data niyo, "Ay, latest 'tong study oh, 2026." So meaning baka mga Gen Z na nagpo-post nito, mache-check natin itong model na 'to, nache-check niya how they construct the Taglish. Kasi nga may mga ginagamit na bagong words eh, kung ang point mo mache-check lang natin 'yung words na pino-post nila.

**Mr. Lu:** I would like to first clarify. From your understanding, you kept mentioning the flooding. What our study does specifically is to test on the Taglish typhoon posts.

**Mr. Peralta:** Mhm.

**Mr. Lu:** And why we chose 2020 to 2026, as mentioned by my fellow researcher, that posts from previous times that are much older may not have the same sentence structure as the newer ones. Not to mention that we will only be taking into consideration the posts from the tropical cyclones that have at least 400 posts after cleaning. If they do not have at least 400 posts, we will not be considering them. And then also, for the API, I believe that it needs to be further explained by my fellow researcher, but from your understanding, we will be personally talking via something similar to an email, is that what your understanding is? Please explain further how the API processes because from their understanding...

**Mr. Peralta:** Hindi, medyo hesitant ako doon sa may pagbabago. May study na ba na nangyari before na napatunayan na noong 2022 to 2026 may pagbabago the way we post?

**Mr. Ducyogen:** Kay Seeberger po, pero it's not a local study.

**Mr. Peralta:** Ang focus niya ay nagbabago ang post—ibang bansa? Hindi ang Taglish na language, ibang bansa. So sa bansa nila, ang study niya, nag-iiba the way 'yung sentence construction noong before sa ngayon. So ibig sabihin may iba nang tinuturo sa klase kung paano tayo gumamit ng language na ito, so napatunayan ba na ganoon? Na-gets niyo ba 'yung idea? Kasi ang sabi niyo nga kasi may nag-research na ang post daw before, nag-iba na sa post ngayon. So meaning we're talking about 'yung sentence. Dati ang sentence medyo Jejemon, 'di ba? Kung Taglish tayo. Ngayon, as time goes by, iba na, medyo clear na. And then 2026 may mga bagong words, 'di ba? Ang daming trending na words. So kung ang target niyo naman ay may pagbabago, may study ba dito sa Pilipinas na ma-justify natin na may pagbabago the way we construct, especially Taglish na posting?

**Mr. Peralta:** Kasi kagaya niyan, Taglish, kami Bisaya kami, hindi kami maalam sa Tagalog, iba 'yung pag-post namin ng Tagalog and the English. So may napatunayan ba kayo? Kaya nga gusto ko ma-justify niyo bakit 2022 to 2026, kung pwede naman tayong mag-focus ng isang buwan sa Pilipinas na nandoon 'yung sakuna na 'yan tapos hanapin natin 'yung mga Taglish na post. And then isa pang problem niyan, paano natin madi-divide nang maayos na sa 2022, ito 'yung number na kukunin namin, sa 2023 ito 'yung number, 2024 ito 'yung number? So divided 'yun sa lahat kasi mahaba 'yung year na kinuha niyo eh. Whereas kung mag-focus tayo sa specific na may sakuna na buwan, which is justifiable kasi sa Pilipinas, 'yan 'yung buwan na may sakuna, 'di ba? So paano natin 'yun idi-divide 'yung pagkuha niyo ng dataset niyo na post, which is the post itself? Kasi kung ibang study dapat kung atin i-study natin kung may kaibahan ba 'yung posting, iju-justify niyo kasi 'yun eh, kung may kaibahan sa pagpo-post ng ano, dati ba kaunting post lang na ganito, "baha dito," o ganyan, o 'di ba? So may pagbabago, ibang study 'yon para i-justify natin na there are changes in the way they post. Pero kung ang gusto niyo lang naman is ma-interpret 'yung Taglish, hindi na issue 'yon. Dapat mag-focus tayo doon on a certain year, on a certain month, or three months. Maganda noong 2025 kasi doon nag-usbong 'yung sa flood control eh. Specific, June to kailan 'yung nagkaroon ng ganoon, ang daming post noon, so doon kayo mag-focus. "The study focuses on June to ganito, ganyan," para isa lang ang problema niyo, hindi na tayo maghahanap ng ibang years pa. Do you get my point? Pinapahirapan niyo lang din 'yung sarili niyo, 2022, hindi niyo naman ma-justify bakit tayo nagsimula ng 2022. O sige, Sir.

**Mr. Bautista:** Kasi from public posts on X, your AI is trying to identify a few things, 'di ba, sa Taglish post. So you're trying to identify what? The type of emergency, number one. The place of the emergency, 'di ba, number two. The people involved, the needed supplies, and lastly, since it's Taglish, which is the point ni Sir, the nuances of the language. So kasi the language, there's difference of use depending if you're in the South, if you're in Manila, and also the time frame. Today, the use of the language of Taglish is not the same today as five years ago. So you are saying maybe that's why you chose 2022 to something. Does your model require a certain number of posts to be able to train properly? Are you saying those four years you're studying has about 100 posts, that's why your language model needs those 100 posts to be able to train properly? Like different combinations ng use of Taglish, use of the place? Or you just chose four years?

**Mr. Bautista:** Now, if you're saying, sabi ni Sir, you just want that model to learn the current use, just use the data ngayon. So what's your justification or your logic for using that dataset? Is it a requirement of your model, it needs all those combinations ng Taglish, date, location, bla-bla-bla to be able to properly optimize? Or is the current data enough to be able to come up with an accurate translation? So that's why when you're doing a dataset, there has to be a justification. Why do you need 10 years of data as opposed to five years of data? One year of data against 10 years of data. If you're saying your AI needs 10 years or five years or four years, why? Different type of language? So in those four years, you'll come up with, four years ago, this was the language, how Taglish was used, now it's different, that's why you need four years. Or is it just enough that, as your MVP, you just need to capture current information for the language to be able to translate properly?

**Mr. Bautista:** So why did you choose that dataset? I think that's the question of Sir. Why did you choose that dataset, four years? If it's easier just to use this month, sabihin mo na lang, "This is the typhoon period lang this year." Is that enough for your model to be able to optimize? Is that enough, or do you just need four years? Do you need four years, or this year is just enough?

**Mr. Ducyogen:** This year is just enough po.

**Mr. Bautista:** This year is just enough?

**Mr. Ducyogen:** Our primary reason was it was a cost-saving measure po to get the...

**Mr. Bautista:** Cost-saving, four years?

**Mr. Ducyogen:** From the older datasets that were not ano yet with the payment.

**Mr. Bautista:** So you see where our questions are leading to. Usually we ask why did you choose this number, you justify why. Why 13, why four years of data? There has to be a reason behind that. So you're trying to optimize, so why not two, why not this month only, typhoon period? So if you give your reason, it should be backed up, because you're saying maybe those four years had 20 bad typhoons, that's why we need that kind of data. So just take note of that and then we can move on. We don't want an answer now. If you don't have any answer on that now, we can move on, just take note of that.

**Mr. Peralta:** Talk with your group kung paano niyo ma-justify, kung nag-agree na ba kayo kung sapat na ba 'yung ngayong taon na makuha niyo 'yung mga ganoong data, iba't ibang klase ng sakuna, 'yung mga binanggit ni Sir. 'Pag hindi, 'tsaka kayo mag-add ng isa pang taon, na "We added 2025 kasi mas marami." Na-identify sa Pilipinas may bagyo, may lindol, may kung ano-ano pa, so 'yung taon na 'yon sapat na for our model to be trained properly.

**Mr. Bautista:** Or like what you said kanina, 'yung ano, use of the language.

**Mr. Peralta:** Language. Pero kung hindi naman ganoon 'yung sinasagot ng SOP niyo na malaman kung may kaibahan ba 'yung pag-post, ang pinaka-objective lang naman is ma-translate itong Taglish na 'to, I don't think na kailangan pa 'yon. That's a separate study na kung ang AI niyo ay mag-a-identify kung may kaibahan ba ang pag-construct ng sentence noong mga year na 'yon. Kasi kung may kaibahan, ibig sabihin may mga additional factors na nag-i-insert bakit nag-iiba. Sa social media ba, sa mga vloggers ba, kaya ganoon na magsalita ang generation ngayon? So may mapapatunayan pa, that's another study. So hindi naman ganoon 'yung study niyo, 'di ba? Sa AI naman, to train lang naman the AI to do 'yung mga sinabi niyo kanina. So pag-decide kayo. Masyadong mahaba 'yung 2022 to 2026. Pero kung ma-justify niyo 'yon, why not? Ilagay niyo rito. Tatanungin na lang 'yan sa susunod kung ma-justify. And then hahanapin 'yon dito sa SOP niyo, kasi ipapresent niyo 'yon na ito 'yung naging resulta, and then you can create a table na magko-compare, 2022 ito 'yon, 2023, so nakita siya. Medyo matrabaho, so ang dami niyong table na ilalagay. Pero 'pag nagkasundo kayo, so 'yon. Okay? O sige, move on tayo.

**Mr. Ducyogen:** For the scope of our study, 3,900 posts will be labeled for both tasks. That number came from Tabuzo et al., which suggested that the minimum amount was 2,800 so that there is no bias. The number 3,900 is the upper range. That number is only used for NER, and we will be using it for the two tasks. Earlier Philippine studies focused on one model for each task, but our study will use one model for both tasks.

**Mr. Ducyogen:** We will use also one encoder for every optimizer, we will use the same dataset, and we will use the same training process so that the comparison is fair. We will also use a prototype with features...

**Mr. Peralta:** May I clarify that 3,900 posts? Is that covered for your study, or nag-mention ka lang ng researcher na nag-mention noon?

**Mr. Ducyogen:** Covered for our study po.

**Mr. Peralta:** Covered for your study. And then naka-base din 'yan doon sa nag-research or 'yung na-reference niyo, tama?

**Mr. Ducyogen:** Yes, Ma'am.

**Mr. Peralta:** Sa kanya 3,900, may specific ba doon na bakit ginaya niyo siya na 3,900? Six years ang kinuha niyo eh, so 3,900 posts. What kind of posts is this? Is there a specific calamity? Kasi 'di ba sabi hindi lang typhoon, tama ba, or specific to typhoon lang?

**Mr. Ducyogen:** Typhoon lang po.

**Mr. Peralta:** Sa typhoon. 'Yung 3,900 na 'yon, is it divided, even siya sa lahat? Kasi may South, kagaya sa amin, iba kami magsalita ng Tagalog eh. Iba magsalita ang taga-Butuan, taga-Davao. Kasi Taglish ang inaano niyo eh. So meron bang ganoon sa pagpili ng ganoon? Na-specify ba dito sa study niyo na about 3,900 posts from—i-mention niyo 'yung nagsabi ng ganoon, and then iaano niyo anong specific lang na area dito sa Pilipinas ang kukuhanin no'n. And then kasi medyo malayo 'yon, 'no, 2022 to 2026, which is babaguhin na natin, 'di ba, kasi sabi ko nga pag-usapan niyo. Lalo na 'yan, 3,900. Tapos ilan sa Visayas, ilan sa Mindanao, ilan sa Luzon na post? Alangan namang Luzon lang i-focus niyo, hindi maganda 'yung data, unless 'yun 'yung nasa study niyo, na "The study focuses only on Luzon, and then specific parts of Luzon," kung tatlo ba ang kinukuhan niyo sa Luzon, 'di ba? Kasi iba-iba magsalita. Iba kasi ang training ng AI niyo kung pure Tagalog 'yung nagsasalita eh, iba ang posting sa X, tama Sir, 'di ba? Nandito nga ako sa Manila pero Bisaya ako, iba din ako mag-post. So that's why you need to justify that 3,900 posts. Assuming na 3,900 posts, magbabayad kayo, magkano 'yon, Sir, sa...

**Mr. Cruz:** 0.01 dollars.

**Mr. Peralta:** Dollar per post. Tapos 'yun na nga sinasabi ko, lalo na 2022 to 2026, mas mahal siguro 'pag older kasi hahanapin pa nila sa database nila 'yon. That's why nag-recommend ako na latest na lang, itong 2026, tapos anuhin niyo na lang anong specific na buwan, tapos specific na lugar sa Pilipinas. Kasi maghahanap ka pa, Cebuano, lalo Cebuano iba din magsalita, Batangueño iba mag-Tagalog.

**Mr. Ducyogen:** Ang covered lang po namin is Taglish language, we will not cover other languages po.

**Mr. Peralta:** Oo. Hindi, I mean kasi the way we speak Tagalog, 'yung pag-construct namin, iba. Nababaliktad namin. Kunwari ang sentence construction sa Tagalog, mauna dapat ito tapos ito, nababaliktad namin 'yun sometimes. Kasi iba kami mag-ano, pero nagta-Tagalog kami. So ipo-post namin, hindi niyo naman ma-identify sa post na 'yon kung taga-saan, kung ano 'yung mother tongue niya. So nagpo-post lang siya, nakita ng AI niyo na, "Ah, this is Taglish." So kasama 'yun sa dataset niyo, 'di ba? So that's why I'm asking you. Hindi naman ibang language, Taglish. Gumamit ako ng Tagalog pero Bisaya ako, iba ako mag-post ng Tagalog. Nakita niyo naman 'yung mga artista 'pag nagsalita, 'pag Bisaya, Sir, 'di ba naiiba 'yung Tagalog, kaya sabi nila, "Mali 'yung Tagalog mo, ito 'yung tamang Tagalog." So Taglish pa rin 'yon.

**Mr. Ducyogen:** That's because we used Herrera's filter po, where any—

**Mr. Peralta:** 'Yun nga, that's why you need to specify that in your document. Ito 'yung year, and then specific na 'yung kukunin na data, kasi kasama 'yan sa scope and limitation niyo, na ito lang ang sakop ng data namin doon sa population. I don't know lang sa CS, anong part 'yun, Ma'am Florentino, ng document? Is it the sample study, sample data? 'Yung number of posts nila, Ma'am, na training data. Sa scope, Sir, 'di ba? Ime-mention niyo 'yan, 'yung 3,900, pero you need to justify bakit 3,900, kaninong study, is it already proven na the standard is 3,900 to be able for your model to be trained properly. So 'yun 'yon. And then i-mention niyo kung anong specific na lugar sa Pilipinas ang kukunin niyo. Ilagay niyo doon sa scope ninyo, nakalagay na Metro Manila, or NCR lang, kasi malaki naman 'yung NCR. So nakatulong 'yun sa inyo kasi na-filter niyo na, "Okay, hanap tayo sa X na..." Kasi kung buong database ang ipapahalungkat niyo sa X, baka hindi lang ganoon ang ipapabayad, kasi nga ang dami noong data. 'Di ba may database subject naman kayo, alam niyo naman ang data doon. So kung naka-specific, na-filter na natin, NCR, 3,900 posts, lahat for NCR lang. So ibig sabihin, 'yung Taglish natin, hindi na natin iniisip kung paano mag-post ng Taglish 'yung iba, 'di ba? Kasi ibang study na 'yon, ibang AI na 'yung kayang mag-train. Naalala ko kasi, Ma'am, may seminar kami noong nakaraan, may bagong AI, 'yung Babaylan. Iba siya sumagot 'pag Binisaya, 'pag sa South, 'yung sa Korea. Noong nakaraang Monday lang, nagpa-seminar kasi ang DICT. Kaya na-mention ko, ibang AI 'yon kasi iba 'yung pagbabasa niya.

**Mr. Peralta:** So kung tatanggapin niyo 'yung suggestion namin, mag-specific na kayo ng lugar, tapos specific din na year, kung gusto niyo year or buwan, na kukunin niyo 'yung mga post. Mas maganda kasi hindi naman kailangan masabi na talagang maganda 'yung resulta niyo, kasi wala naman kayong pinapatunayan. Ang pinaka-ano natin sa study niyo, anong magiging resulta gamit 'yung mga model na pinili niyo. You don't need to justify or to confirm na may napatunayan kayo para lang ma-impress 'yung magbabasa nito. Hindi ganoon ang research. Ibig sabihin, "Sa year na ganito, ito 'yung resulta." Wala tayong pakialam kung anong resulta, pinapa-inform lang natin sa public na ito ang resulta. Hindi niyo kailangan i-perfect na gumana 'yung model na 'to. Kaya nga meron tayong research gap, 'di ba? 'Yun 'yung lalabas doon. Magkasundo kayo, mahal 'yan, 3,900 posts 'yan eh.

**Mr. Bautista:** That's actually the use of the scope, 'di ba? In scope, out of scope. You're just building an MVP. You do not need to anticipate all the combinations, all the situations. You just limit it to a particular one, like sabi ni Sir, NCR, Taglish of NCR only. Nothing wrong with that. You're just limiting it to your MVP. You're not building the whole application. It's MVP only. "For this study, the scope is the Taglish of NCR." Kasi you're opening doors for questions. "Oh, paano 'yung use of Taglish in the South, and why all these combinations of years?" So that's for a different purpose or a different study. Just limit your study to a certain scope so it will be easier for you. That's only MVP. Later on, siguro you can add those features, so it can identify all types of Taglish in the Philippines. Pero for now, for your study, the scope is Taglish NCR. Naintindihan?

**Mr. Bautista:** You may not need all those 3,900 posts, kasi 3,900, different types of Taglish na 'yun, different situations. So spend some time on the scope. "This is only what we are researching and presenting." So you will not be opening lots of questions, because you're anticipating all the situations, and that's not what we're asking. Limit that, it's very open. Taglish posts from 2022 to 2026, why? Mainly from X. Additionally, be prepared for what will happen after translation. Accuracy. How does your system handle the accuracy of the translation, the accuracy of the source of information? Is it true? Sabi ni Sir kanina, baka gawa-gawa lang 'yon. So be prepared later on how you will handle those situations, accuracy of translation and accuracy of the information that you're getting. Okay? Sir Joel, Sir, meron pa?

**Mr. Cruz:** Una na 'ko. Kanina pa ako nag-iisip. What would be the use or application of this study? Ano 'yung magiging pakinabang natin dito? Kunwari it works at the end. For what purpose siya? 'Yun ang iniisip ko kanina pa.

**Mr. Florentino:** Proceed to the benefits. Who's going to discuss that?

**Mr. Almuena:** There's three main benefactors sa study namin. Firstly, 'yung mga LGU disaster offices, because we're able to give them a single model that can handle multiple types of tasks, like the report types and key details, all while being able to run locally without having to use a paid LLM service like Claude, Gemini, whatever. Meaning, they'll be able to run it on their own hardware rather than having to pay a separate entity. As for local universities, we want to give them specifics on how our 13 multi-task optimizers are able to reduce negative transfer, and at the same time the training cost, for pre-trained encoders and how they handle multiple tasks. And finally, in terms of communities, as mentioned earlier, we want first responders to use our model so that they're able to sort through and filter out the information, just so they'll be able to react faster.

**Mr. Cruz:** Again, why do you have to filter?

**Mr. Ducyogen:** The reason we need to filter is that not all posts are useful po. Some posts are trolls, some are noisy, some are not a call for help.

**Mr. Cruz:** So going back doon sa na-mention ko kanina pa, iko-correct niya 'yung message? Is it correcting the message, 'yung post, the way 'yung pagkakasabi sa post tulad kanina?

**Mr. Lu:** No, we are not trying to correct the post. What we are trying to do is determine what is the intent of the post po.

**Mr. Cruz:** What is the intent?

**Mr. Lu:** Such as whether it is asking for help, where it is asking for help, who are the ones that need help, and what they need.

**Mr. Cruz:** For example, paano kung nagkamali rin 'yung AI in interpreting 'yung mga data na 'yon? How can you be sure na at least 80 to 100%? Kasi kung nagkamali rin ng interpretation 'yung AI, meron bang contingency plan or...

**Mr. Peralta:** I think, Sir, magte-test naman sila, 'di ba, kasi may number of posts naman kayo. So I think may gagamitin naman sila doon sa treatment ng data nila, and then i-justify niyo kung ilang percentage ng result 'yung pasado para masabi na ito ay acceptable na. So I think they do have that in the treatment of data.

**Mr. Lu:** We do have a part from the 3,900 posts. We will take a part of it as a testing dataset to determine the accuracy, and then we will be making a decision on which optimizer to keep afterwards based on that.

**Mr. Peralta:** May specific ba kayo na percentage para masabi na when it comes to speed, sa accuracy, ito 'yung pasado? Meron bang ganoon? Kasi 'yun 'yung sinasabi ni Sir, paano kung nagkamali 'yan. Normal talaga na meron ganoon, wala namang 100% na result. So ano 'yung acceptable na percentage? Tatlo ang mine-measure niyo doon, accuracy, speed, and may pangatlo sa SOP niyo kanina, nakalimutan ko 'yung isa. 'Yun 'yung kine-question ni Sir, paano 'pag nagkamali.

**Mr. Cruz:** During testing niyo, are you planning to get 'yung mga data na you think malaki 'yung possibility na magkaroon ng error doon sa result, para ma-compare kung na-detect ba talaga na may problema or what?

**Mr. Ducyogen:** We are classifying it po, classifying on the meaning—

**Mr. Peralta:** I think hindi part ng study nila, Sir, to verify.

**Mr. Bautista:** 'Yun 'yung study. That's the study.

**Mr. Peralta:** Ah, that's the study na?

**Mr. Cruz:** To verify kung—

**Mr. Peralta:** To verify kung tama ba 'yung post, hindi lang siya basta post.

**Mr. Bautista:** How do you know the accuracy of the post? That's step number one. There's a post. How does your system or the model know this post is true? 'Yun 'yung first, before all the translation, kasi after translation you will check the accuracy also. First step, how do you know this post is true and you really need to get it? Do you cross-check that with another database, or do you just agree it's true because it's on X?

**Mr. Florentino:** May part ng testing doon, Sir. They will explain.

**Mr. Peralta:** May assurance ba na 'pag binili natin kay X, ang ibibigay nila ay 'yung legit na post, or kayo pa ang magsasala doon since bayad naman siya? Para maka-ano tayo doon sa question ni Sir, para masabing fake 'to na post. Baka kasi 'yung mga paulit-ulit lang na post. Kasi may mga troll, 'di ba, 'yung mga binabayaran na troll farm. Tapos iisang tao lang pala nag-post no'n, iniiba lang para masira itong lugar na 'yon, "Baha dito, tapos sira ang ano," 'di ba? Tapos dito naman, siya pa rin 'yon, ibang account lang, nagda-dummy account. So kasama na ba 'yun? Or may dinadaanan tayo to check that? Pwede naman kasi ilagay 'yan sa conclusion niyo, na ito 'yung percentage na hindi accurate, or certain level ng accuracy. Meron bang ganoon?

**Mr. Bautista:** Or is that out of scope? Out of scope ng research, you just take that as 100% true. Lahat ng nakikita niyo, you take it as 100% true, you do not need to validate this post. Are you saying that, or are you also validating this post for truthfulness?

**Mr. Ducyogen:** Out of scope po.

**Mr. Bautista:** Out of scope. Is it in your paper that it's out of scope, that you're accepting...? Is it clear?

**Mr. Peralta:** Ime-mention niyo 'yan para clear siya na out of scope. And then paano niyo ite-test 'yung accuracy? What do you mean by accuracy? Accuracy para ma-identify itong post na ito? Is that what you mean by accuracy? Or accuracy ng pag-translate noong post na 'yon? Nasa SOP 3 'yun, I think, tatlo SOP niyo. 'Yung accuracy ba ng pag-translate and speed ng pag-translate?

**Mr. Florentino:** You start from there, Mr. Lu. Explain how you're going to do it.

**Mr. Ducyogen:** First, the post will be cleaned. By cleaning, we will remove the personal details and redundant characters po. After that, we will label the post. By labeling, we will put the expected correct answer. For example, that post is an urgent request. This is done by a human.

**Mr. Bautista:** I'm sorry, it's done by a human? 'Yung labeling?

**Mr. Ducyogen:** Yes, it's not done by a machine or AI.

**Mr. Bautista:** Oh, there's manual intervention in labeling?

**Mr. Cruz:** So siya 'yung nagde-decide kung—

**Mr. Florentino:** Machine learning po.

**Mr. Bautista:** For training the model.

**Mr. Florentino:** Oo, sa training.

**Mr. Ducyogen:** The labeled dataset will then be used to train the model. After that, we will train the single-task models and the 13 candidate models. We will save those models, then we will do the optimizer selection. During the optimizer selection, we will evaluate which of those 13 candidate models will be used for the final model, which will be used for the prototype application.

**Mr. Bautista:** How would you know the labels are correct?

**Mr. Ducyogen:** By using Cohen's kappa score, which will be presented in the next part.

**Mr. Lu:** The labeling must be accurate and trustable. To measure this, we will be having two annotators and measuring how much they agree on the same posts in the same batch. We aim to have at least a Cohen's kappa of 0.61 for categories and an F1 score of 0.74 for entities. In the time that either of these targets is not met, we will be re-evaluating the labeling guidelines and revising them before the next batch.

**Mr. Bautista:** Okay, very good answer. Follow-up lang. You have two annotators, 'di ba, independent? What happens if they disagree with each other?

**Mr. Lu:** If they disagree to an extent that falls below the target, we will be re-evaluating the guidelines and revising them before the next batch po.

**Mr. Bautista:** Can you go back there? So you have two independent annotators labeling, and what happens if there's a disagreement? You mentioned the guideline. So one annotator says, "This should be the label," and the second one says, "No, this should be the label." Based on the guideline, how will they resolve those disagreements? Who will prevail over the other? You have two, 'di ba, and they're both independent in their labeling.

**Mr. Ducyogen:** In the data gathering, we have a third annotator that will—

**Mr. Bautista:** Is there a lead and a secondary? Primary versus secondary, or are they of equal weight? If they are of equal weight and there's a disagreement, who will say which one is correct?

**Mr. Ducyogen:** There is a third annotator po.

**Mr. Bautista:** Ah, there's a third annotator, not in this slide. Because here it only says two independent annotators, they label following a guideline. So in your guideline, can you show what the resolution will be if there's a disagreement? Or you can say you're assuming they will not disagree.

**Mr. Ducyogen:** There will be disagreements po, but that disagreement will be handled by a third annotator.

**Mr. Bautista:** How will they resolve that disagreement? Is it in your—

**Mr. Ducyogen:** Yes po, it's in the methodology, under data gathering.

**Mr. Bautista:** Sorry, can you show that in the guideline? "If two annotators disagree on labeling, this is the resolution." Because that will happen. That will always happen when there are two people with different backgrounds, different mindsets. One will say this, the other will sometimes say this. Or are you assuming again that will not happen, "For this study, we assume they will always agree on the labeling"? Or you're saying, "Later on, we will be introducing a third annotator to resolve any disagreements. For this study, there's only two, and we'll assume there's no disagreement."

**Mr. Peralta:** I think, guys, kasi medyo ano na 'yung oras niyo. 'Di ba may binanggit kayong dalawang annotators, and then merong isa na idadagdag. I think you need to specify that in your document. Number one, since in-introduce niyo 'yung dalawang annotator, i-specify niyo ang classification ng annotator na 'to. Kasi hindi pwedeng kung sino-sino na lang ang kukunin mong annotator. You need to have an expert. Anong qualification ng expert na 'to? Programmer ba ito? Hindi basta-basta. And also set a timeline. So mag-add kayo ng table specifying the classification of the annotators and the specification of the agreement and the disagreement. Kasi may clause dapat doon na 'pag hindi nagkasundo 'yung dalawa, magdadagdag ng another annotator, its qualification, and then 'yung rules niyo para mag-come up with the final decision. Kasi hindi pwedeng naglagay ka lang doon ng "annotator," kasi nga study 'to, so dapat naka-reflect 'yon. So you will be adding that here in your document, 'yung table of specification para sa annotators and the guidelines. Okay?

**Mr. Peralta:** Baka kasi kumuha lang kayo dito. Dapat 'yung nasa industry, talagang programmer, hindi 'yung basta lang may nakita kayo. Kagaya ko, nakita niyo ako, ako pa ang tiningnan, hindi naman ako CS graduate, 'di ba? So dapat may specification, and then papirmahin talaga natin na tiningnan niya talaga 'yung labels. Oo, so 'yun ang gusto ni Sir, tapos 'yung specific na guidelines para sa pangatlong annotator. Kasi hindi pwedeng dalawa sila, Sir, iba-iba ang sagot, tapos ang pangatlo, paano siya magde-decide? So dapat may specific guidelines kung paano magde-decide itong pangatlong annotator.

**Mr. Lu:** After this, each model will then be tested for its accuracy. To do this, we will take one task as an example, which is named entity recognition. What the model does is it goes word by word and tries to find the key details. For example, here. If these answers are compared to the answer key, which we already know for this specific test, we will be able to see that some of its labels are correct, some of them are wrong, and it might miss some key details. This is done for every post.

**Mr. Lu:** And then take this situation for example. We might have an eager model. It marks almost every single word as something important. And if it marks almost every word, of course, it will find the key details. The issue is that many of the words that it marks are wrong, resulting in a low F1 score or low accuracy. The opposite is also the same, where a careful model marks only what it is sure is correct. What it does is it has a high accuracy. The issue is that it might miss many of the key details, also receiving a low F1 score. In order to receive a high F1 score, it must balance both things, which is to have correct answers and also find most or all of the correct answers.

**Mr. Bautista:** Sorry, just to clarify, when you say accuracy, accuracy of translation?

**Mr. Lu:** No, the accuracy of finding the key details for named entity recognition, the second task po.

**Mr. Bautista:** Yeah, the translation from the Taglish to get the details, the accuracy of that? Like bagyo, typhoon. Baha, flood. Something like that?

**Mr. Lu:** Ah, yes po.

**Mr. Bautista:** Do you also check the urgency of that? Kasi this is calamities. Do you also consider the urgency, let's say water shortage versus an earthquake? Later on, do you consider that in terms of sending out the message, like this is more important, for this particular scope?

**Mr. Lu:** The urgency po is actually a different task—

**Mr. Bautista:** Or you just treat every calamity as the same? Baha, water shortage, lindol, fire.

**Mr. Ducyogen:** For the labels po, we will use the labels from a prior research, which was Alam et al., wherein those are the main labels—

**Mr. Bautista:** Like if two X posts come up, one is a fire in one barangay, against no water in two barangays. Which one will the model work on? Priority, urgency, or you do not consider it? That's okay as well.

**Mr. Ducyogen:** For this, it's not considered po.

**Mr. Bautista:** You should put that as part of the out of scope as well. Because sometimes there's a big post like, "Oh no, nag-break si ganyan at si ganyan. Nagkakagulo na 'yung mga tao rito kasi nag-break 'yung dalawang artista," versus there's a fire in a barangay, and you treat those posts as the same. Or again, you can say, "For this project, we do not classify the disasters or calamities." Please add that as well.

**Mr. Ducyogen:** We do not classify the disasters po. We added one on the scope, that we do not consider other hazards.

**Mr. Bautista:** Then another explicit out of scope for the type of hazards.

**Mr. Ducyogen:** Type of hazards na rin po?

**Mr. Bautista:** Yeah, because some will say, "Nagkakagulo na 'yung mga tao rito kasi nag-break 'yung love team na 'to, may riot na," against there's a fire. So may riot. It will pick up, "Oh, it's a riot in this barangay," but the context of the post is a love team breaking up. Pero sasabihin niya riot, that's a calamity as well. So it may pick that up. So that also goes back to the accuracy of the translation. Is it a disaster, a love team, just a trash post?

**Mr. Ducyogen:** Out of the scope po, nakalagay doon sa—

**Mr. Bautista:** And the model should be able to distinguish. It's not a trash post, it's not a—

**Mr. Ducyogen:** Hindi na po kasama sa dataset.

**Mr. Bautista:** Oo, pero there's a part of cleaning—

**Mr. Florentino:** I-clarify mo nga 'yung tasks. What are the tasks that your ano will do?

**Mr. Ducyogen:** 'Yung task lang po is classification of the intent of the post according to the—

**Mr. Bautista:** It also boils down to the truthfulness of the post. Is it accurate? Pero where do you validate that? Hindi. For now, you just accept whatever that is.

**Mr. Ducyogen:** Yes po.

**Mr. Lu:** This score is then taken into consideration for the multi-task model. To get the negative transfer, we take the score of the multi-task model and subtract it by the score of the single-task model. If the multi-task model has a lower score than the single-task model, that means it has less accuracy, and it has negative transfer.

**Mr. Lu:** We will also be measuring the speed, which is the amount of time from the first training post that enters the model up to the time that the model is finished and saved. We will also be measuring the time to label each post, which is the time from when the post enters the model until the labeling output comes out. Next, we will be measuring the peak memory, which is the highest point of memory it reached during training. We will also be measuring the size of the saved model.

**Mr. Lu:** Finally, we will take the optimizers and determine whether they have any difference in their negative transfer at all. If they do, a Dunnett's test will then be used to compare the optimizers to equal weighting, to determine which optimizers have significantly less negative transfer compared to equal weighting. Those that qualify are only the ones that also do not negatively impact the accuracy of either task. From those that qualify, we will be taking the optimizer that has the least amount of extra training cost.

**Mr. Bautista:** Okay. So I think this is your criteria for choosing the optimizers. That should have been your answer earlier. "We chose these 13 optimizers because..." Or is this just choosing the particular optimizer? The 13 you have not answered. Here you're choosing the particular optimizer because of this. It is faster, it passes, so it's the final optimizer. So this is your criteria for choosing. Did I get that right or not?

**Mr. Ducyogen:** Yes, this is the criteria for choosing the final—

**Mr. Bautista:** Particular optimizer. Okay. And there's a question there, "Do the 13 differ at all?"

**Mr. Ducyogen:** In terms of reducing the negative transfer po.

**Mr. Bautista:** Ah, 'yun nga. Okay.

**Mr. Lu:** And then if none qualify, we will instead be choosing equal weighting, as this is the optimizer that has the least amount of extra training cost. Finally, we will be taking the final model with the final optimizer that we chose earlier. We will be testing them on the exact same test posts across five random seeds. This is to ensure that each model has one run that is equal to the run of another.

**Mr. Lu:** We will then be using a paired t-test to determine the average gap of each run away from zero. And then we will check whether they have a clear difference. The clear difference is determined when the range includes zero. If zero is included, that means that the multi-task model had no clear difference from the single-task model. If it is a positive number and does not include zero, that means that the multi-task model improved compared to the single-task model. If the entire range falls below zero, which is a negative number, that means that the single-task model performed better.

**Mr. Lu:** Now for the use case test. Say, for example, for creating a label, we will take a sample input and create the expected result. If the actual result matches the expected result, we will consider the use case as passed.

**Mr. Bautista:** How do you measure urgency?

**Mr. Ducyogen:** We will not measure urgency po, because it is outside of the scope.

**Mr. Bautista:** Urgency, no. Kasi you said there, "request or urgent needs." How do you differentiate a request from urgent needs?

**Mr. Ducyogen:** In terms of, for example, "We need a rescue." For example lang po, "Kailangan po namin ng rescue sa Barangay Legazpi." We will consider that as an urgent need, as rescue can be a form of urgent needs.

**Mr. Bautista:** How did it determine it's urgent?

**Mr. Ducyogen:** Are you thinking po of urgency in terms of urgency levels?

**Mr. Bautista:** Yeah. What's a request, what's an urgent need for your—

**Mr. Ducyogen:** So a request is, for example, "Kailangan po namin ng tubig." 'Yun lang po.

**Mr. Bautista:** Okay, and how urgent is that?

**Mr. Ducyogen:** We will not measure the urgency level po. Our definition for now of urgent needs is depending on the severity of the—

**Mr. Bautista:** Or the AI will just define that? You're saying the AI knows what's urgent and what's a request?

**Mr. Ducyogen:** We will define it in the labeled dataset po.

**Mr. Bautista:** Okay, so we can find here how you label? The guideline for labeling is here?

**Mr. Ducyogen:** For now, we only put the labels there po.

**Mr. Bautista:** What's the guideline? Request, tubig, or clothes, or remove obstruction on the road. It could wait. Urgent is it cannot wait. So how do you train your model on that? How will it know if this is only a request that can wait for hours or days, or it's an urgent need, meaning there's a fire? How will the model know that? Or will it just treat it as first in, first out, or last in, first out, let's say it's a stack? How will the model know na it's only a request and it is not urgent? How do you train the—you have a blank AI, it doesn't know what a request or an urgent need is. Or do you already have an intelligent AI that knows that?

**Mr. Ducyogen:** No po, we don't have a guideline doon sa pag-classify—

**Mr. Bautista:** Yeah, it's still not part of your scope. Okay.

**Mr. Ducyogen:** Yes po.

**Mr. Patricio:** This Gantt chart shows the proposed timeline of the study. After this defense, Chapters 1 to 3 will be revised according to the panel's feedback. On mid-November, we will start collecting the posts and prepare them for labeling. Labeling will run from December to late January. We will not wait for the labeling to finish. While the dataset is still being labeled, we will start building the benchmark code and its tests, along with the prototype and its tests. From mid-January to late February, we will start training the model, and as the results come in, we will start to write Chapters 4 and 5.

**Mr. Bautista:** What's Chapter 4 and 5?

**Mr. Florentino:** 1 to 3 lang, Sir.

**Mr. Bautista:** Kasi nakalagay "write Chapters 4 and 5."

**Mr. Florentino:** Yes. 'Yun na 'yung parang conclusion, Sir, data analysis.

**Mr. Bautista:** I just want to know, if the data is coming, what's 4 and 5.

**Mr. Patricio:** So all tasks end before March, and March is free because it serves as a buffer in case of any delays.

**Mr. Almuena:** So here we're going to have a training pilot for the 13 optimizers, ensuring that they all fall under 8 gigabytes. We're going to have a fixed workload of 92 tuning runs with 51 main runs.

**Mr. Geronimo:** As soon as the proposal is accepted, we will revise the chapters based on the panel's feedback. After that, we will secure the X API for archive search, since most of our posts will come from X.

**Mr. Ducyogen:** Running alongside the pilot search, we will begin drafting the guidelines for the annotators that will be annotating the batches. We will be setting up doccano as a labeling tool.

**Mr. Bautista:** Do you have a test plan? A test plan shows all the test scripts that you will do, the test procedure, the actual result.

**Mr. Peralta:** Kasi magpa-pilot testing kayo before niyo ma-ano 'yung system niyo, 'di ba, 'yung app niyo, kasi gagawa kayo. Meron ba kayong test plan?

**Mr. Bautista:** Details, test script, test script ID, expected result, actual result. So do you have that plan? At this point, none yet. You just have a timeline, "Our testing, based on our test plan, from this to this." Pero no details pa.

**Mr. Ducyogen:** Yes po, kasi we will be—

**Mr. Peralta:** Oo, kasi dapat nandito na 'yun, kasi Chapter 1 to 3 'to. Ito kasing document na 'to ang magga-guide on how you proceed, guide niyo na 'yun. 'Di ba ite-test niyo 'yung data? So dapat meron kayong test plan, 'di ba? Ite-test niyo 'yung negative transfer na sinasabi niyo. So may ginagamit kayo, 'yung F1 score, nakita ko 'yun, gumamit kayo ng F1 score sa accuracy. Merong formula 'yon, 'yung true positive, true negative, false positive. So idadagdag niyo 'yon. And then dapat meron kayo no'n, 'yung document na 'yon. Anong test script, ito ba 'yung pag-label? Oo, tapos ano 'yung result?

**Mr. Bautista:** Who's handling testing pala sa inyo? Who does testing?

**Mr. Lu:** Testing in what sense po? Can you clarify po?

**Mr. Peralta:** 'Yung mga test script niyo.

**Mr. Bautista:** 'Yung test plan, your test approach, what's your test strategy. It goes back to what dataset you are going to use and why, up to the timeline, up to the actual test scripts na i-e-execute ng tester.

**Mr. Florentino:** Sir, at this point, ang concentration nila is on the research. 'Yung software engineering part, they will be deciding on it once you approve the... Pero they will include the test plan in Chapter 3, kasi kailangan nasa 3 'yun, pero wala pa siya rito. Pero you mentioned 'yung formulas ninyo for testing, 'di ba?

**Mr. Peralta:** F1, nakita ko may F1 score.

**Mr. Lu:** We're using the F1 score po for the accuracy ng dalawang tasks.

**Mr. Peralta:** True positive, oo. So they should include that, F1 score. Sige na, move on na tayo. Lagpas na tayo sa 1 hour.

**Mr. Ducyogen:** The annotators will then be labeling a practice batch with the first draft of the guidelines that we have created. Let's say the posts from Twitter or X are not enough, we will then take the public Facebook posts. The other condition is if the annotators do not agree on the practice batch. If they do not agree, we will revise the guidelines. Now, if both conditions are true, we will then proceed to full labeling as early as December.

**Mr. Lu:** For technical risks and backup plans, we have identified four main risks. First is code bugs. Of course, there might be issues with the parts of our research. We will be testing every single part, which also includes the optimizers, by using certain inputs where we already know the correct output. We will then be running a small sample of posts through the entire pipeline.

**Mr. Lu:** We have also identified the hardware limit, which we identified during our training run. Here we were able to see that four optimizers actually needed more memory than our 8 GB graphics processing unit could accommodate. To accommodate for this, we will be freezing the embedding layer of every model.

**Mr. Lu:** For the prototype failure, we will be testing each use case again. If it fails the use case, where the actual result does not match what we already know to be correct, we will be fixing it until it gives the correct answer and passes.

**Mr. Bautista:** See, that's also you talking about your testing plan, testing strategy, or testing procedure. You're already testing. You're testing the use case, and if there's an error, fix it. So you are testing. Actually, you have a test process already in place, you are just not mentioning your whole strategy and the details. You already have it in there. So the details of how you would test the use case, that's part of the test plan. "This is the scope of the test plan, we will test this, these are the actual test scripts we're going to use." We do not expect all those details here, just mention the strategy and the approach. Kaya I'm asking kanina, what's in Chapter 4 and 5? Is that going to be it? That's why I kept asking kanina. So here you're already mentioning testing.

**Mr. Florentino:** So you want it included, Sir, in Chapter 3, the test plan?

**Mr. Bautista:** Oo, just a high level, what's in scope. That includes nga 'yung dataset kanina.
