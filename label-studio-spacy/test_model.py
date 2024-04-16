import spacy

# Load the trained SpaCy NER model
nlp = spacy.load("/home/debo/Rayserver/label-studio-spacy/ner_model_server/output/model-best")

# Sample texts for testing
texts = [
    """J U D G M E N T CIVIL APPEAL NO. 5795 OF 2007 (Arising out of SLP (C) No. 8041 of 2007) Dr. ARIJIT PASAYAT, J.

1. Leave granted.

2. Challenge in this appeal is to the order passed by the High Court of Gujarat dismissing the writ petitions filed by the appellant.

3. Challenge before the High Court was to the order dated 10.1.2006 passed by the Customs, Excise & Service Tax Appellate Tribunal (in short 'CESTAT') directing deposit of rupees two crores as a condition precedent for entertaining the appeal. It is to be noted that the total amount of penalty imposed was Rs.10,00,00,000/-. The dispute relates to classification of the product imported by the appellant and consequential benefits claimed by it under various Notifications issued by the Director General of Foreign Trade. The customs authorities did not accept the stand of the appellant about its classification. The levy of penalty was challenged by way of appeal before the CESTAT. It was accompanied by an application seeking waiver of the penalty imposed by the Commissioner of Customs (in short the 'Commissioner').

4. After hearing the parties, the CESTAT inter alia noted as follows:

"The learned Advocate for the applicants contends in one hand that a letter of Ms. Indian Oil Corporation Ltd. written by its Assistant Manager, was on the record of the Commissioner in these proceedings and was not considered in spite of the directions in remand to consider all materials. It was also submitted that the directions of the DGFT dated 17.12.1997 have not been complied, with even though certificates showing the use of the return stream were on record in parallel proceedings before the department. It was also submitted by the learned advocate that if these certificates were considered, then they should be granted the benefit of DGFT waiver of condition of the resale of the return stream, vide their letter dated 17.12.1997 and they were not so liable to any penalty. The Learned advocate took us through the Balance Sheet of the applicant company which discloses that as on 31.3.2005, they have a loss of Rs.12.20 crores and in the earlier year the said loss was Rs.17.74 crores. He submits that they are a BIFR company and pleads for full waiver of the pre-deposit requirement under Section 129E of the Customs Act to hear this appeal. The Learned SDR on the other hand takes us through the letter dated 17.12.1997 of DGFT and submits that this letter exempts and is applicable only to import of naphtha and return stream of such naphtha. He submits that the letter relied upon by the advocate of Indian Oil Corporation, which he is making a grievance about, it stating that heptene is not known and understood as naphtha. The certificates of consumption of the return stream are also certifying the utilization and the return stream of nonene and heptene and not to naphtha.
Considering the submission in this matter, prima facie we are of the view that the waiver granted of the condition by the DGFT is not applicable to the subject imports in this case. The other issues raised will have to be gone into in detail at the regular hearing. At this prima facie stage considering the merits and the financial position as also the fact that this is the second round, we would consider this case to be appropriate to direct the applicants to terms of pre-deposit requirement to be effected under Section 129E of the Customs Act, 1962. We would, therefore, direct the applicants to deposit Rs.2,00,00,000/- (rupees two crores only) and report compliance thereof within 12 (twelve) weeks i.e. on 17.4.2006. On such compliance, being reported, the matter would be listed for regular hearing. Failure to deposit and meet the requirement of Section 129E calls for dismissal of the appeal without any further notice."
5. The penalty, it is to be noted, has been imposed under Section 112 (a) of the Customs Act, 1962 (in short the 'Act'). A specific finding was recorded by the Commissioner that in respect of the goods the assessee was required to obtain a licence for clearance. However, the value of the goods which could have been confiscated was in the neighbourhood of Rs.66.57 crores. As it was not possible to direct confiscation since they were released to the appellant for use in the factory premises, the Commissioner imposed penalty of Rs.10,00,00,000/-.

6. The matter was earlier before the Tribunal and at that stage matter had been remitted for fresh adjudication. By order dated 11.4.1998 the Commissioner passed a fresh order and the levy of penalty of Rs.10,00,00, 000/- was re-affirmed.

7. Learned counsel for the appellant submitted that the bona fide of the appellant is writ large. The company has become a sick company and, therefore, insistence on pre deposit even of a part which is in this case a huge sum of rupees two crores would deprive the appellant of the statutory right of appeal. It is pointed out that from the financial statements it is clear that the appellant has suffered huge losses. For the assessment years 31st March, 2004, 31st March, 2005 and 31st March, 2006 respectively the figures of losses are Rs.17.74 crores, Rs.12.20 crores and Rs.8.28 crores. It is stated that the financial position has become dismal and insistence on pre-deposit is by overlooking the financial sickness of the company. The imports in question were made during the period 1992 to 1997. There was lot of confusion and because of that dispute has arisen. Reference is made to certain communications of the DGFT and Indian Oil Corporation (in short 'IOC').

8. Learned counsel for the respondents on the other hand has submitted that there is no prima facie case and even if it is conceded for the sake of arguments that there is financial hardship, that cannot be a ground to dispense with pre deposit and in any event the balance of convenience is not in favour of the appellant.

9. We shall deal with first the issue relating to the question of stay/dispensation of pre deposit in respect of sick industry. In Metal Box India Ltd. v. Commissioner of Central Excise, Mumbai (2003 (155) ELT 13 (S.C.), this Court had clearly observed as follows:

"Mr. Rana Mukherjee, the learned counsel for the appellants submits that in view of Section 22 of the Sick Industrial Companies (Special Provisions) Act, 1985 (for short 'the Sick Industries Act'), the appellant need not deposit the amount, as ordered by the Tribunal, as protection is available to the appellant under the said provision. We are afraid, we cannot accept the contention of the learned counsel for reasons more than one. First, this aspect was not the subject matter of the order under challenge and, secondly, Section 22 of the Sick Industries Act provides relief in regard to the proceedings which relate to (a) winding up of the industrial company; (b) execution distress or the like against any of the properties of the industrial company; (c) the appointment of a receiver in respect thereof, and (d) proceeding in regard to suit for recovery of money or for the enforcement of any security against the industrial company or of any guarantee in respect of any loans or advance granted to the industrial company. Payment of pre-deposit covered under Section 35F of the Central Excise Tax Act, 1944 does not fall under any of the above-mentioned categories in Section 22 of the Sick Industries Act."
10. Principles relating to grant of stay pending disposal of the matters before the concerned forums have been considered in several cases. It is to be noted that in such matters though discretion is available, the same has to be exercised judicially.

11. The applicable principles have been set out succinctly in Silliguri Municipality and Ors. v. Amalendu Das and Ors. (AIR 1984 SC 653), M/s Samarias Trading Co. Pvt. Ltd. v. S. Samuel and Ors. (AIR 1985 SC 61) and Assistant Collector of Central Excise v. Dunlop India Ltd. (AIR 1985 SC 330).

12. It is true that on merely establishing a prima facie case, interim order of protection should not be passed. But if on a cursory glance it appears that the demand raised has no leg to stand, it would be undesirable to require the assessee to pay full or substantive part of the demand. Petitions for stay should not be disposed of in a routine matter unmindful of the consequences flowing from the order requiring the assessee to deposit full or part of the demand. There can be no rule of universal application in such matters and the order has to be passed keeping in view the factual scenario involved. Merely because this Court has indicated the principles that does not give a license to the forum/authority to pass an order which cannot be sustained on the touchstone of fairness, legality and public interest. Where denial of interim relief may lead to public mischief, grave irreparable private injury or shake a citizens' faith in the impartiality of public administration, interim relief can be given.

13. Section 129-E of the Act reads as follows:

"129E. DEPOSIT, PENDING APPEAL, OF DUTY AND INTEREST DEMANDED OR PENALTY LEVIED. - Where in any appeal under this Chapter, the decision or order appealed against relates to any duty and interest demanded in respect of goods which are not under the control of the customs authorities or any penalty levied under this Act, the person desirous of appealing against such decision or order shall, pending the appeal, deposit with the proper officer the duty and interest demanded or the penalty levied. Provided that where in any particular case, the Commissioner (Appeals) or the Appellate Tribunal is of opinion that the deposit of duty and interest demanded or penalty levied would cause undue hardship to such person, the Commissioner (Appeals) or, as the case may be, the Appellate Tribunal may dispense with such deposit subject to such conditions as he or it may deem fit to impose so as to safeguard the interests of revenue."
14. Two significant expressions used in the provisions are "undue hardship to such person" and "safeguard the interests of revenue". Therefore, while dealing with the application twin requirements of considerations i.e. consideration of undue hardship aspect and imposition of conditions to safeguard the interest of Revenue have to be kept in view.

15. As noted above there are two important expressions in Section 129-E. One is undue hardship. This is a matter within the special knowledge of the applicant for waiver and has to be established by him. A mere assertion about undue hardship would not be sufficient. It was noted by this Court in S. Vasudeva v. State of Karnataka and Ors. (AIR 1994 SC

923) that under Indian conditions expression "Undue hardship" is normally related to economic hardship. "Undue" which means something which is not merited by the conduct of the claimant, or is very much disproportionate to it. Undue hardship is caused when the hardship is not warranted by the circumstances.

16. For a hardship to be 'undue' it must be shown that the particular burden to have to observe or perform the requirement is out of proportion to the nature of the requirement itself, and the benefit which the applicant would derive from compliance with it.

17. The above position has been highlighted in M/s Benara Valves Ltd. and Ors. v. Commissioner of Central Excise and Anr. (2006 (12) SCALE 303). Though the said case related to dispute under the Customs Excise Act, 1944 (in short the 'Excise Act') the parameters are the same.

18. We do not find any infirmity in the order directing deposit of Rupees two crores as affirmed by the High Court. The appellant is granted three months time to deposit the amount fixed by the CESTAT. If it is not deposited within the aforesaid time, the appeal before the CESTAT shall stand dismissed.

19. The appeal is disposed of accordingly with no order as to costs.
    """
]

# Process each text and print the entities
for text in texts:
    doc = nlp(text)
    # print("Text:", text)
    for ent in doc.ents:
        print(f"Entity: {ent.text}, Label: {ent.label_}")
    print()
